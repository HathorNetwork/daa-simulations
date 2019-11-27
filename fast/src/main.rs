use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, LinkedList};
use std::error::Error;
use std::ffi::OsStr;
use std::fmt;
use std::fmt::Debug;
use std::fs::File;

use env_logger::Env;
use log::LevelFilter;
use log::{error, info, trace, warn};
use maplit::*;
use noisy_float::prelude::*;
use petgraph::prelude::*;
use plotters::coord::Shift;
use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Standard};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::path::PathBuf;
use structopt::StructOpt;

type BoxedDaa = Box<dyn Daa>;
type BoxedMiner = Box<dyn Miner>;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Config {
    target_solvetime: u32,
    initial_weight: f64,
    min_weight: f64,
    weight_decay: bool,
    daa: BoxedDaa,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            target_solvetime: 30,
            initial_weight: 50.,
            min_weight: 21.,
            weight_decay: true,
            daa: Box::new(<Htr as Default>::default()),
        }
    }
}

impl Config {
    fn target_solvetime_log2(&self) -> f64 {
        (self.target_solvetime as f64).log2()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum Event {
    MineBlock {
        delay: f64,
        block: BlockWithMetadata,
        mining_job_uid: u64,
    },
    PropagateBlock(BlockWithMetadata),
    TakeMeasurements,
}

impl Event {
    fn delay(&self) -> Option<f64> {
        if let &Event::MineBlock { delay, .. } = self {
            Some(delay)
        } else {
            None
        }
    }
}

struct DaaContext<'a> {
    node: &'a Node,
    config: &'a Config,
    known_blocks: &'a HashMap<Hash, BlockWithMetadata>,
}

#[typetag::serde]
trait Daa: Debug + objekt::Clone + Send + Sync {
    fn next_weight(&self, ctx: DaaContext) -> R64;
}
objekt::clone_trait_object!(Daa);

trait LogAdd<Rhs = Self>
where
    Self: std::marker::Sized,
{
    type Output;
    #[must_use]
    fn log_add(self, rhs: Rhs, base: f64) -> Self::Output;
    #[must_use]
    fn log2_add(self, rhs: Rhs) -> Self::Output {
        self.log_add(rhs, 2.0f64)
    }
}

trait LogSub<Rhs = Self>
where
    Self: std::marker::Sized,
{
    type Output;
    #[must_use]
    fn log_sub(self, rhs: Rhs, base: f64) -> Self::Output;
    #[must_use]
    fn log2_sub(self, rhs: Rhs) -> Self::Output {
        self.log_sub(rhs, 2.0f64)
    }
}

impl LogAdd for f64 {
    type Output = f64;
    fn log_add(self, rhs: f64, base: f64) -> f64 {
        let a = self.max(rhs);
        let b = rhs.min(self);
        a + base.powf(b - a).ln_1p() / base.ln()
    }
}

impl LogAdd for R64 {
    type Output = R64;
    fn log_add(self, rhs: R64, base: f64) -> R64 {
        let a = self.max(rhs);
        let b = rhs.min(self);
        a + r64(base).powf(b - a).ln_1p() / r64(base).ln()
    }
}

impl LogSub for R64 {
    type Output = R64;
    fn log_sub(self, rhs: R64, base: f64) -> R64 {
        let a = self.max(rhs);
        let b = rhs.min(self);
        a + (-r64(base).powf(b - a)).ln_1p() / r64(base).ln()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ScheduledEvent {
    uid: u64,
    node_idx: Option<NodeIndex>,
    timestamp: R64,
    event: Event,
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp.cmp(&other.timestamp).reverse()
    }
}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.uid == other.uid
    }
}

impl Eq for ScheduledEvent {}

#[derive(Serialize, Deserialize, Copy, Clone, Eq, PartialEq, Hash)]
struct Hash(u64);

impl Hash {
    fn null() -> Hash {
        Hash(0)
    }
}

impl Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hx{:016x}", self.0)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq, Hash)]
struct Block {
    parent_hash: Hash,
    timestamp: u32,
    rweight: R64,
    // consider using target bits instead of weight
    //target_bits: u32,
    nonce: u32,
}

impl Block {
    /// Use Rust's DefaultHasher to generate the block hash
    fn hash(&self) -> Hash {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        // this is needed because there is a name collision: Hash::hash and Self::hash
        <Self as Hash>::hash(self, &mut hasher);
        Hash(hasher.finish())
    }
    fn genesis(config: &Config) -> Block {
        Block {
            parent_hash: Hash::null(),
            timestamp: 0,
            rweight: r64(config.initial_weight),
            nonce: 0,
        }
    }
    fn weight(&self) -> f64 {
        self.rweight.raw()
    }
}

// This is separated to make it easier to impl Hash+Eq on Block
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct BlockMetadata {
    hash: Hash,
    height: u64,
    solvetime: u32,
    logwork: R64,
    mined_by: NodeIndex,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct BlockWithMetadata {
    block: Block,
    metadata: BlockMetadata,
}

impl BlockWithMetadata {
    fn genesis(config: &Config) -> BlockWithMetadata {
        BlockWithMetadata {
            block: Block::genesis(config),
            metadata: BlockMetadata {
                hash: Hash::null(),
                height: 0,
                solvetime: 0,
                logwork: r64(config.initial_weight),
                mined_by: 0.into(),
            },
        }
    }

    /// iteretate over the best chain, starting from the current block
    fn iter_blockchain<'a>(
        &self,
        known_blocks: &'a HashMap<Hash, BlockWithMetadata>,
    ) -> impl Iterator<Item = &'a BlockWithMetadata> {
        IterChain {
            known_blocks,
            current_hash: self.metadata.hash,
        }
    }
}

#[derive(Debug)]
struct IterChain<'a> {
    known_blocks: &'a HashMap<Hash, BlockWithMetadata>,
    current_hash: Hash,
}

impl<'a> std::iter::Iterator for IterChain<'a> {
    type Item = &'a BlockWithMetadata;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_hash == Hash::null() {
            None
        } else if let Some(block) = self.known_blocks.get(&self.current_hash) {
            self.current_hash = block.block.parent_hash;
            Some(block)
        } else {
            error!("blockchain ended before reaching genesis");
            None
        }
    }
}

#[test]
fn test_genesis_eq_itself() {
    let config = Default::default();
    assert_eq!(Block::genesis(&config), Block::genesis(&config));
    assert_eq!(
        BlockWithMetadata::genesis(&config),
        BlockWithMetadata::genesis(&config)
    );
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Edge {
    rtt: f64,
    sigma: f64,
}

impl Edge {
    fn new(rtt: f64, sigma: f64) -> Edge {
        Edge { rtt, sigma }
    }
    fn sample_delay<R: Rng>(&self, rng: &mut R) -> f64 {
        Normal::new(self.rtt, self.sigma)
            .unwrap()
            .sample(rng)
            .max(0.001f64) // min 1ms
    }
}

impl Default for Edge {
    fn default() -> Self {
        Edge::new(0.050, 0.010)
    }
}

#[typetag::serde]
trait Miner: Debug + objekt::Clone + Send + Sync {
    // H/s TODO: use PH/s for more real world values
    fn hashrate(&mut self) -> f64 {
        self.hashrate_peek()
    }
    fn hashrate_peek(&self) -> f64;
}
objekt::clone_trait_object!(Miner);

#[derive(Debug, Clone)]
struct NodeTemplate {
    name_template: &'static str,
    miner: Option<BoxedMiner>,
}

impl NodeTemplate {
    fn new(name_template: &'static str, miner: Option<BoxedMiner>) -> NodeTemplate {
        NodeTemplate {
            name_template,
            miner,
        }
    }
    fn create_node(&self, node_idx: u32, config: &Config) -> Node {
        Node {
            idx: Default::default(),
            name: format!("{} {}", self.name_template, node_idx),
            miner: self.miner.clone(),
            mining_job_uid: 0,
            timestamp: 0.0,
            tip: BlockWithMetadata::genesis(config),
            blocks_seen: hashset! {Hash::null()},
        }
    }
}

fn sample_solvetime<R: Rng>(
    rng: &mut R,
    hashrate: f64,
    rweight: R64,
    target_solvetime: u32,
) -> u32 {
    use rand::distributions::Open01;
    // sample how many tries are needed to solve at the current difficulty weight
    let uniform: f64 = rng.sample(Open01);
    //let trials = 2.0f64.powf(rweight.raw()) * -uniform.ln();
    let geom_p = 2.0f64.powf(-rweight.raw());
    let trials = uniform.ln() / (-geom_p).ln_1p();
    (target_solvetime as f64 * trials / hashrate) as u32
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Node {
    name: String,
    idx: NodeIndex,
    miner: Option<BoxedMiner>,
    mining_job_uid: u64,
    timestamp: f64,
    tip: BlockWithMetadata,
    blocks_seen: HashSet<Hash>,
}

impl Node {
    fn height(&self) -> u64 {
        self.tip.metadata.height
    }

    fn create_mining_job(
        &mut self,
        config: &Config,
        known_blocks: &mut HashMap<Hash, BlockWithMetadata>,
    ) -> Option<Event> {
        if let Some(ref mut miner) = self.miner {
            let hashrate = miner.hashrate();
            let mut rng = thread_rng();
            let new_block = self.gen_new_block(&mut rng, hashrate, config, self.idx, known_blocks);
            known_blocks.insert(new_block.metadata.hash, new_block.clone());
            let delay = new_block.metadata.solvetime as f64;
            Some(Event::MineBlock {
                delay: delay,
                block: new_block,
                mining_job_uid: self.mining_job_uid,
            })
        } else {
            None
        }
    }

    fn on_block_found(
        &mut self,
        block: BlockWithMetadata,
        config: &Config,
        propagated: bool,
        known_blocks: &mut HashMap<Hash, BlockWithMetadata>,
    ) -> LinkedList<Event> {
        let mut list = LinkedList::new();
        // ignore known blocks
        if self.blocks_seen.contains(&block.metadata.hash) {
            return list;
        }
        self.blocks_seen.insert(block.metadata.hash);
        // invalidate current mining job
        self.mining_job_uid += 1;
        // if blocks arrive out of order, the following assumptions may lead to a fork
        if !propagated {
            trace!("new block: {:?}", block);
        }
        if block.metadata.logwork > self.tip.metadata.logwork {
            self.tip = block.clone();
        }
        list.push_back(Event::PropagateBlock(block));
        list.extend(self.create_mining_job(config, known_blocks));
        list
    }

    fn gen_new_block<R: Rng>(
        &self,
        rng: &mut R,
        hashrate: f64,
        config: &Config,
        node_idx: NodeIndex,
        known_blocks: &HashMap<Hash, BlockWithMetadata>,
    ) -> BlockWithMetadata {
        let mut rweight = config.daa.next_weight(DaaContext {
            node: &self,
            config: config,
            known_blocks: known_blocks,
        });
        let mut solvetime = sample_solvetime(rng, hashrate, rweight, config.target_solvetime);
        if config.weight_decay {
            let mut max_k = 300;
            while solvetime > max_k * config.target_solvetime {
                rweight -= r64(2.73); // 15% of original diff
                solvetime = max_k * config.target_solvetime
                    + sample_solvetime(rng, hashrate, rweight, config.target_solvetime);
                max_k += 60;
            }
        }
        let rweight = rweight;
        let solvetime = solvetime.max(1);

        // there is at least the genesis block, unwrapping won't panic
        let parent_block = &self.tip;
        let block = Block {
            parent_hash: parent_block.metadata.hash,
            // timestamp at least 1s higher than parent block
            timestamp: (self.timestamp as u32) + solvetime,
            rweight: rweight,
            nonce: rng.sample(Standard),
        };
        let hash = block.hash();
        //let solvetime = block.timestamp - parent_block.block.timestamp;
        let block_with_metadata = BlockWithMetadata {
            block,
            metadata: BlockMetadata {
                hash,
                height: parent_block.metadata.height + 1,
                solvetime,
                logwork: parent_block.metadata.logwork.log2_add(rweight),
                mined_by: node_idx,
            },
        };
        block_with_metadata
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ConstantMiner {
    hashrate: f64,
}

impl ConstantMiner {
    fn new(hashrate: f64) -> Box<ConstantMiner> {
        Box::new(ConstantMiner { hashrate })
    }
}

#[typetag::serde]
impl Miner for ConstantMiner {
    fn hashrate_peek(&self) -> f64 {
        self.hashrate
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct VariableMiner {
    cur_hashrate: f64,
    multiplier: f64,
    /// 0.0 means equal chance of rising and falling, 1.0 means always rising
    rise_bias: f64,
}

impl VariableMiner {
    fn new(cur_hashrate: f64, multiplier: f64, rise_bias: f64) -> Box<VariableMiner> {
        Box::new(VariableMiner {
            cur_hashrate,
            multiplier,
            rise_bias,
        })
    }
}

#[typetag::serde]
impl Miner for VariableMiner {
    fn hashrate(&mut self) -> f64 {
        self.cur_hashrate
    }
    fn hashrate_peek(&self) -> f64 {
        self.cur_hashrate
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
//struct Network(StableUnGraph<Node, Edge>);
struct Network(UnGraph<Node, Edge>);

impl Network {
    fn node(&self, node_idx: NodeIndex) -> Option<&Node> {
        self.0.node_weight(node_idx)
    }

    fn nodes(&self) -> impl Iterator<Item = &Node> {
        use petgraph::visit::IntoNodeReferences;
        self.0.node_references().map(|(_, n)| n)
    }

    fn nodes_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.0.node_weights_mut()
    }

    fn node_mut(&mut self, node_idx: NodeIndex) -> Option<&mut Node> {
        self.0.node_weight_mut(node_idx)
    }

    fn remove_node(&mut self, node_idx: NodeIndex) -> Option<Node> {
        self.0.remove_node(node_idx)
    }

    fn add_connected_node(&mut self, node: Node) -> NodeIndex {
        let node_idx = self.0.add_node(node);
        let mut node = self.0.node_weight_mut(node_idx).unwrap();
        node.idx = node_idx;
        // collect so we can drop `self.network` read ref and mut it below
        let neighbors: Vec<_> = self
            .0
            .node_indices()
            .filter(|&idx| idx != node_idx)
            .collect();
        for idx in neighbors.iter() {
            self.0.add_edge(node_idx, *idx, Default::default());
        }
        node_idx
    }

    fn edges(&self, node_idx: NodeIndex) -> impl Iterator<Item = (NodeIndex, &Edge)> {
        self.0.edges(node_idx).map(|e| (e.target(), e.weight()))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Measurement {
    estimated_network_weight: f64,
    best_height: u64,
    orphan_blocks: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Simulator {
    config: Config,
    timestamp: f64,
    next_event_uid: u64,
    next_node_uid: u32,
    scheduled_events: BinaryHeap<ScheduledEvent>,
    network: Network,
    max_steps: u64,
    known_blocks: HashMap<Hash, BlockWithMetadata>,
    initial_timestamp: f64,
    measurements: Vec<Measurement>,
    measurement_period: f64,
}

impl Simulator {
    fn new(config: Config) -> Simulator {
        let genesis_block = BlockWithMetadata::genesis(&config);
        let mut scheduled_events = BinaryHeap::new();
        scheduled_events.push(ScheduledEvent {
            uid: 0,
            node_idx: None,
            timestamp: r64(0.0),
            event: Event::TakeMeasurements,
        });
        Simulator {
            config,
            timestamp: 0.0,
            next_event_uid: 1,
            next_node_uid: 0,
            scheduled_events,
            network: Default::default(),
            //history: vec![],
            max_steps: 1_000_000, // arbitrary
            known_blocks: hashmap! {
                Hash::null() => genesis_block,
            },
            initial_timestamp: 0.0,
            measurements: vec![],
            measurement_period: 1.0,
        }
    }

    fn iter_measurements(&self) -> impl Iterator<Item = (f64, &Measurement)> {
        struct StepBy {
            cur: f64,
            step: f64,
        }
        impl StepBy {
            fn new(start: f64, step: f64) -> StepBy {
                StepBy { cur: start, step }
            }
        }
        impl Iterator for StepBy {
            type Item = f64;
            fn next(&mut self) -> Option<Self::Item> {
                let ret = self.cur;
                self.cur += self.step;
                Some(ret)
            }
        }
        StepBy::new(self.initial_timestamp, self.measurement_period).zip(self.measurements.iter())
    }

    fn example_node(&self) -> Option<&Node> {
        use std::convert::identity;
        self.network
            .0
            .node_indices()
            .next()
            .map(move |node_idx| self.network.node(node_idx))
            .and_then(identity)
    }

    fn create_node(&mut self, template: &NodeTemplate) -> NodeIndex {
        let node = template.create_node(self.next_node_uid, &self.config);
        self.next_node_uid += 1;
        self.add_node(node)
    }

    fn add_node(&mut self, mut node: Node) -> NodeIndex {
        let mining_job = node.create_mining_job(&self.config, &mut self.known_blocks);
        let node_idx = self.network.add_connected_node(node);
        if let Some(event) = mining_job {
            self.schedule_event(Some(node_idx), event);
        }
        node_idx
    }

    fn remove_node(&mut self, node_idx: NodeIndex) -> Option<Node> {
        self.network.remove_node(node_idx)
    }

    fn schedule_all_events(
        &mut self,
        node_idx: Option<NodeIndex>,
        events: impl IntoIterator<Item = Event>,
    ) {
        for schedule_event in events {
            self.schedule_event(node_idx, schedule_event);
        }
    }

    fn schedule_event(&mut self, node_idx: Option<NodeIndex>, event: Event) {
        let timestamp = r64(self.timestamp);
        let mut rng = thread_rng();
        if let Some(delay) = event.delay() {
            self.scheduled_events.push(ScheduledEvent {
                uid: self.next_event_uid,
                node_idx,
                timestamp: timestamp + delay,
                event,
            });
            self.next_event_uid += 1;
        } else if let Some(node_idx) = node_idx {
            for (idx, edge) in self.network.edges(node_idx) {
                let delay = edge.sample_delay(&mut rng);
                self.scheduled_events.push(ScheduledEvent {
                    uid: self.next_event_uid,
                    node_idx: Some(idx),
                    timestamp: timestamp + delay,
                    event: event.clone(),
                });
                self.next_event_uid += 1;
            }
        } else {
            unreachable!()
        }
    }

    fn run(&mut self, timeout: f64) {
        let start_timestamp = self.timestamp;
        let mut steps = 0u64;
        loop {
            let max_steps = self.max_steps;
            // abort the simulation if it exceeds the max number of steps
            if steps > max_steps {
                warn!("simulation stalled");
                break;
            }
            steps += 1;
            // peek to see if we have to stop
            if let Some(scheduled_event) = self.scheduled_events.peek() {
                if scheduled_event.timestamp - start_timestamp > timeout {
                    trace!("reached max simulation time");
                    break;
                }
            } else {
                trace!("no more events");
                break;
            }
            // actually pop the next event
            if let Some(ScheduledEvent {
                node_idx,
                timestamp,
                event,
                ..
            }) = self.scheduled_events.pop()
            {
                trace!("run event {:?}", event);
                self.timestamp = timestamp.raw();
                let network = &mut self.network;
                let mut some_node = node_idx.and_then(|idx| network.node_mut(idx));
                // trace!("node {:?} not found, skipping event {:?}", node_idx, event);
                // continue;
                if let Some(ref mut node) = some_node {
                    node.timestamp = self.timestamp;
                }
                match event {
                    Event::MineBlock {
                        block,
                        mining_job_uid,
                        ..
                    } => {
                        let node = if let Some(node) = some_node {
                            node
                        } else {
                            trace!(
                                "node {:?} not found, Event::MineBlock requires node",
                                node_idx
                            );
                            continue;
                        };
                        if node.mining_job_uid == mining_job_uid {
                            let next_events = node.on_block_found(
                                block,
                                &self.config,
                                false,
                                &mut self.known_blocks,
                            );
                            let node_idx = node.idx;
                            drop(node);
                            drop(network);
                            self.schedule_all_events(Some(node_idx), next_events);
                        }
                    }
                    Event::PropagateBlock(block) => {
                        let node = if let Some(node) = some_node {
                            node
                        } else {
                            trace!(
                                "node {:?} not found, Event::PropagateBlock requires node",
                                node_idx
                            );
                            continue;
                        };
                        let next_events =
                            node.on_block_found(block, &self.config, true, &mut self.known_blocks);
                        let node_idx = node.idx;
                        drop(node);
                        drop(network);
                        self.schedule_all_events(Some(node_idx), next_events);
                    }
                    Event::TakeMeasurements => {
                        // estimate network hashrate
                        let mut iweights = network
                            .nodes()
                            .filter(|node| node.miner.is_some())
                            .map(|node| node.miner.as_ref().unwrap().hashrate_peek().log2());
                        let some_weight = iweights.next();
                        let estimated_network_weight = some_weight
                            .map(|w| iweights.fold(w, |w1, w2| w1.log2_add(w2)))
                            .unwrap_or(0.0);
                        // find best height: height of the best chain among all nodes
                        let best_block = network
                            .nodes()
                            .map(|n| &n.tip)
                            .max_by_key(|b| b.metadata.logwork)
                            .unwrap(); // every node has at least the genesis, unwrap is safe
                        let best_height = best_block.metadata.height;
                        // total number of orphan blocks so far
                        let orphan_blocks = self.known_blocks.len() as u64 - best_height;
                        // save measurement
                        self.measurements.push(Measurement {
                            estimated_network_weight,
                            best_height,
                            orphan_blocks,
                        });
                        self.scheduled_events.push(ScheduledEvent {
                            uid: self.next_event_uid,
                            node_idx: None,
                            timestamp: timestamp + self.measurement_period,
                            event: Event::TakeMeasurements,
                        });
                        self.next_event_uid += 1;
                    }
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ConstantDaa {
    weight: R64,
}

impl Default for ConstantDaa {
    fn default() -> Self {
        ConstantDaa { weight: r64(63.5) }
    }
}

#[typetag::serde]
impl Daa for ConstantDaa {
    fn next_weight(&self, _ctx: DaaContext) -> R64 {
        self.weight
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Htr {
    window_size: u64,
    max_dw: Option<f64>,
}

impl Default for Htr {
    fn default() -> Self {
        Htr {
            window_size: 20,
            max_dw: Some(0.25),
        }
    }
}

#[typetag::serde]
impl Daa for Htr {
    fn next_weight(&self, ctx: DaaContext) -> R64 {
        use std::cmp::min;

        let height = ctx.node.height();
        if height < 3 {
            return r64(ctx.config.initial_weight);
        }

        let n = min(self.window_size, height) as usize;
        let tip = &ctx.node.tip;
        let tail = ctx
            .node
            .tip
            .iter_blockchain(ctx.known_blocks)
            .skip(n - 1)
            .next()
            .unwrap();
        trace!("tip: {:?}, tail: {:?}", tip, tail);
        let dt = (tip.block.timestamp - tail.block.timestamp) as f64;
        let logwork = tip.metadata.logwork.log2_sub(tail.metadata.logwork);
        let weight = logwork - dt.log2() + (ctx.config.target_solvetime as f64).log2();

        // adjust max change in weight if configured
        if let Some(max_dw) = self.max_dw {
            let max_dw = r64(max_dw);
            let old_weight = tip.block.rweight;
            let dw = weight - old_weight;
            if dw > max_dw {
                old_weight + max_dw
            } else if dw < -max_dw {
                old_weight - max_dw
            } else {
                weight
            }
        } else {
            weight
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SimpExp {
    k: f64,
    h: f64,
}

impl Default for SimpExp {
    fn default() -> Self {
        use std::f64::consts::LN_2;
        SimpExp {
            k: 0.01 / LN_2 as f64,
            h: 0.0,
        }
    }
}

#[typetag::serde]
impl Daa for SimpExp {
    fn next_weight(&self, ctx: DaaContext) -> R64 {
        let tip = &ctx.node.tip;
        if tip.metadata.height < 1 {
            return r64(ctx.config.initial_weight);
        }
        let height_from_genesis = self.h + tip.metadata.height as f64;
        let time_since_genesis = tip.block.timestamp as f64;
        let target_solvetime = ctx.config.target_solvetime as f64;
        let next_weight = self.k * (height_from_genesis - time_since_genesis / target_solvetime);
        let next_weight = next_weight.max(ctx.config.min_weight);
        r64(next_weight)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Lwma {
    window_size: u64,
    harmonic: bool,
    //tl_rules: Option<(u64, u64)>,
}

impl Default for Lwma {
    fn default() -> Self {
        Lwma {
            window_size: 134,
            harmonic: true,
            //tl_rules: Some(300, 300)
        }
    }
}

#[typetag::serde]
impl Daa for Lwma {
    fn next_weight(&self, ctx: DaaContext) -> R64 {
        use std::cmp::min;
        let height = ctx.node.height();
        if height < 3 {
            return r64(ctx.config.initial_weight);
        }
        let n = min(self.window_size, height - 1) as usize;
        let k = 2.0f64 / ((n * (n + 1)) as f64);
        let mut lwma_solvetimes = 0.0f64;
        let mut log_sum_inv_weight: Option<f64> = None;
        let mut log_sum_weight: Option<f64> = None;
        let solvetimes_and_weights = ctx
            .node
            .tip
            .iter_blockchain(ctx.known_blocks)
            .take(n)
            .map(|b| (b.metadata.solvetime, b.block.weight()));
        for (i, (solvetime, weight)) in solvetimes_and_weights.enumerate() {
            //let solvetime = if let Some((ptl, ftl)) = self.tl_rules { min(ptl as i64, max(solvetime as i64, -(ftl as i64))) } else { solvetime as i64 };
            lwma_solvetimes += k * (solvetime as f64) * (n - i) as f64;
            log_sum_inv_weight = Some(if let Some(val) = log_sum_inv_weight {
                val.log2_add(-weight)
            } else {
                -weight
            });
            log_sum_weight = Some(if let Some(val) = log_sum_weight {
                val.log2_add(weight)
            } else {
                weight
            });
        }
        let harmonic_mean_weight = (n as f64).log2() - log_sum_inv_weight.unwrap();
        let arithmetic_mean_weight = log_sum_weight.unwrap() - (n as f64).log2();
        let mean_weight = if self.harmonic {
            harmonic_mean_weight
        } else {
            arithmetic_mean_weight
        };
        let next_weight =
            mean_weight + (ctx.config.target_solvetime as f64).log2() - lwma_solvetimes.log2();
        let next_weight = next_weight.max(ctx.config.min_weight);
        r64(next_weight)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Msb {
    window_size: u64,
    sigma_adjust: f64,
    //harmonic: bool,
    //tl_rules: Option<(u64, u64)>,
}

impl Default for Msb {
    fn default() -> Self {
        Msb {
            window_size: 134 * 2,
            sigma_adjust: 5.0,
            //harmonic: true,
            //tl_rules: Some(300, 300)
        }
    }
}

#[typetag::serde]
impl Daa for Msb {
    fn next_weight(&self, ctx: DaaContext) -> R64 {
        use std::cmp::min;
        let height = ctx.node.height();
        if height < 10 {
            return r64(ctx.config.initial_weight);
        }
        let n = min(self.window_size, height - 1) as usize;
        let k = n / 2;
        let kx = (k as f64) / (2 * ctx.config.target_solvetime.pow(2)) as f64;
        let mut sum_diffs = 0.0;
        let mut log_sum_weight: Option<f64> = None;
        let mut sum_solvetimes = 0.0;
        let mut prefix_sum_diffs = vec![0.0];
        let solvetimes = ctx
            .node
            .tip
            .iter_blockchain(ctx.known_blocks)
            .take(n)
            .map(|b| b.metadata.solvetime);
        for solvetime in solvetimes {
            prefix_sum_diffs.push(prefix_sum_diffs.last().unwrap() + solvetime as f64);
        }
        let solvetimes_and_weights = ctx
            .node
            .tip
            .iter_blockchain(ctx.known_blocks)
            .take(k)
            .map(|b| (b.metadata.solvetime, b.block.weight()));
        for (i, (solvetime, weight)) in solvetimes_and_weights.enumerate() {
            let x = (prefix_sum_diffs[i + k] - prefix_sum_diffs[i + 1]) / k as f64;
            let ki = kx * (x - ctx.config.target_solvetime as f64).powi(2);
            let ki = 1.0.max((ki / self.sigma_adjust).ceil());
            if ki > 1.0 {
                trace!("outlier!!! {} {} {}", i, ki, x);
            }
            sum_diffs += ki * 2.0.powf(weight);
            let kweight = weight + ki.log2();
            log_sum_weight = Some(if let Some(val) = log_sum_weight {
                val.log2_add(kweight)
            } else {
                kweight
            });
            sum_solvetimes += ki * solvetime as f64;
        }
        //let next_weight = log_sum_weight.unwrap() - sum_solvetimes.log2() + ctx.config.target_solvetime_log2();
        let next_weight =
            sum_diffs.log2() - sum_solvetimes.log2() + ctx.config.target_solvetime_log2();
        let next_weight = next_weight.max(ctx.config.min_weight);
        r64(next_weight)
    }
}

#[derive(Debug, StructOpt)]
#[structopt(about = "An example of StructOpt usage.")]
struct Opt {
    /// Log level
    #[structopt(long)]
    log: Option<LevelFilter>,
    #[structopt(subcommand)] // Note that we mark a field as a subcommand
    cmd: Cmd,
}

#[derive(Debug, StructOpt)]
#[structopt(about = "the stupid content tracker")]
enum Cmd {
    /// Run the simulation
    Sim(SimOpt),
    /// Plot results
    Plot(PlotOpt),
    /// Convert simulation file
    Convert(ConvertOpt),
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    let mut log_builder = &mut env_logger::from_env(Env::default().default_filter_or("info"));
    if let Some(level) = opt.log {
        log_builder = log_builder.filter_level(level);
    }
    log_builder.try_init()?;
    trace!("{:?}", opt);

    match opt.cmd {
        Cmd::Sim(opt) => main_sim(opt),
        Cmd::Plot(opt) => main_plot(opt),
        Cmd::Convert(opt) => main_convert(opt),
    }
}

#[derive(Debug, StructOpt)]
struct SimOpt {
    /// Possible difficulty adjustment algorithms: "htr", "lwma", "simpexp"
    #[structopt(short, long)]
    daa: String,

    /// Optional JSON file to load simulator states from
    #[structopt(short, long)]
    out_file: Option<PathBuf>,

    /// Append to output instead of replacing
    #[structopt(short, long)]
    append: bool,

    /// Disable weight decay for late blocks
    #[structopt(long)]
    no_decay: bool,

    /// Serialize a pretty JSON
    #[structopt(long)]
    pretty: bool,

    /// Number of parallel simulations to run
    #[structopt(default_value = "1")]
    count: u16,

    /// Simulate for how long
    time: Option<f64>,
}

fn main_sim(opt: SimOpt) -> Result<(), Box<dyn Error>> {
    use std::thread;

    let daa: BoxedDaa = match opt.daa.to_lowercase().as_ref() {
        "const" => Box::new(ConstantDaa {
            ..Default::default()
        }),
        "htr" => Box::new(Htr {
            ..Default::default()
        }),
        "lwma" => Box::new(Lwma {
            ..Default::default()
        }),
        "msb" => Box::new(Msb {
            ..Default::default()
        }),
        "crazy" | "simpexp" => Box::new(SimpExp {
            k: 1. / 134.,
            h: 6000.,
            ..Default::default()
        }),
        other => {
            error!("invalid DAA: {}", other);
            return Ok(());
        }
    };

    let config = Config {
        daa: daa,
        weight_decay: !opt.no_decay,
        ..Default::default()
    };

    let tpl_regular = NodeTemplate::new("node", Some(ConstantMiner::new(1e15)));
    let tpl_strong = NodeTemplate::new("node", Some(ConstantMiner::new(2e18)));
    //let mut sim_threads = vec![];

    let load_sims_handle = if let (&Some(ref path), true) = (&opt.out_file, opt.append) {
        let path = path.clone();
        Some(thread::spawn(move || {
            load_sims(&path).map_err(|e| e.to_string())
        }))
    } else {
        None
    };

    let mut next_sims = vec![];
    (0..opt.count)
        .into_par_iter()
        .map({
            //let config = config.clone();
            let tpl_regular = tpl_regular.clone();
            let tpl_strong = tpl_strong.clone();
            let time = opt.time.unwrap_or(3600. * 24.);
            move |i| {
                let mut sim = Simulator::new(config.clone());
                sim.create_node(&tpl_regular);
                sim.create_node(&tpl_regular);
                sim.create_node(&tpl_regular);
                sim.create_node(&tpl_regular);
                let idx1 = sim.create_node(&tpl_strong);
                let idx2 = sim.create_node(&tpl_strong);
                info!("sim-{} start, run for {}s", i, time);
                sim.run(time);
                let time2 = time * 3.;
                info!("sim-{} remove strong miner, run for {}s", i, time2);
                let _ = sim.remove_node(idx1);
                let _ = sim.remove_node(idx2);
                sim.run(time2);
                info!("sim-{} over", i);
                sim
            }
        })
        .collect_into_vec(&mut next_sims);

    let sims = if let Some(handle) = load_sims_handle {
        let mut sims = handle.join().unwrap()?;
        sims.append(&mut next_sims);
        sims
    } else {
        next_sims
    };

    //for i in 0..opt.count {
    //    let sim_config = config.clone();
    //    let tpl_regular = tpl_regular.clone();
    //    let tpl_strong = tpl_strong.clone();
    //    let time = opt.time.unwrap_or(3600. * 24.);
    //    sim_threads.push(thread::spawn(move || {
    //        let mut sim = Simulator::new(sim_config);
    //        sim.create_node(&tpl_regular);
    //        sim.create_node(&tpl_regular);
    //        let idx = sim.create_node(&tpl_strong);
    //        info!("sim-{} start, run for {}s", i, time);
    //        sim.run(time);
    //        let time2 = time * 6.;
    //        info!("sim-{} remove strong miner, run for {}s", i, time2);
    //        let _ = sim.remove_node(idx);
    //        sim.run(time2);
    //        info!("sim-{} over", i);
    //        sim
    //    }));
    //}

    //// TODO: deal with unwrap
    //sims.extend(sim_threads.into_iter().map(|t| t.join().unwrap()));

    if let Some(ref path) = opt.out_file {
        save_sims(path, &sims, opt.pretty)?;
    }

    Ok(())
}

#[derive(Debug, StructOpt)]
struct PlotOpt {
    /// JSON file to load simulator state from
    #[structopt(long)]
    load_sim: PathBuf,
    /// PNG file to save plot to
    #[structopt(long)]
    save_plot: PathBuf,
    /// When to start plotting
    #[structopt(long)]
    from: Option<u32>,
    /// When to stop plotting
    #[structopt(long)]
    until: Option<u32>,
}

fn main_plot(opt: PlotOpt) -> Result<(), Box<dyn Error>> {
    let path = &opt.save_plot;
    info!("plotting to {:?}", path);
    if path.extension() == Some(OsStr::new("svg")) {
        let mut root = SVGBackend::new(path, (1000, 350)).into_drawing_area();
        _main_plot(&opt, &mut root).unwrap();
    } else {
        let mut root = BitMapBackend::new(path, (2000, 700)).into_drawing_area();
        _main_plot(&opt, &mut root).unwrap();
    }
    info!("plotted");
    Ok(())
}

//fn draw_blockchain<'a, 'b: 'a, DB: DrawingBackend>(&self, root: &'a mut DrawingArea<DB, plotters::coord::Shift>, time_range: impl RangeBounds<u32>) -> Result<(), Box<dyn Error + 'b>>
fn _main_plot<'a, DB: DrawingBackend>(
    opt: &PlotOpt,
    root: &'a mut DrawingArea<DB, Shift>,
) -> Result<(), Box<dyn Error + 'a>> {
    use plotters::palette::{rgb::Rgb, Gradient, LinSrgb};

    let sims = load_sims(&opt.load_sim)?;

    root.fill(&WHITE)?;

    let t0 = 0u32;
    let tf = sims.iter().map(|s| s.timestamp as u32).max().unwrap_or(60);

    let t0 = if let Some(t0) = opt.from { t0 } else { t0 };
    let tf = if let Some(tf) = opt.until { tf } else { tf };

    //let history: vec<_> = self.history.iter().filter(|w| time_range.contains(&w.timestamp)).collect();
    //let max_weight = blocks.iter()
    //    .map(|b| b.weight)
    //    .chain(history.iter().map(|w| w.hashrate_weight(&self.config)))
    //    .fold(0.0f64, |a, b| a.max(b));
    let max_weight = sims
        .iter()
        .map(|s| s.known_blocks.iter().map(|(_, b)| b.block.rweight))
        .flatten()
        // TODO: chain with hashrate history
        .max()
        .map(|r| r.raw())
        .unwrap_or(0.0);
    let min_weight = sims
        .iter()
        .map(|s| s.known_blocks.iter().map(|(_, b)| b.block.rweight))
        .flatten()
        // TODO: chain with hashrate history
        .min()
        .map(|r| r.raw())
        .unwrap_or(0.0);

    let x0 = t0 as f32;
    let xf = tf as f32;
    let y0 = (((min_weight as u64) / 5 - 1) * 5) as f32;
    let yf = (((max_weight as u64) / 5 + 1) * 5) as f32;

    let mut chart = ChartBuilder::on(&root)
        .caption("Blocks difficulty", ("Arial", 20).into_font())
        .margin(3)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_ranged(x0..xf, y0..yf)?;

    chart
        .configure_mesh()
        //.disable_x_mesh()
        //.set_tick_mark_size(LabelAreaPosition::Top, 3600)
        //.set_tick_mark_size(LabelAreaPosition::Bottom, 3600*24)
        .y_desc("Difficulty (log2 scale)")
        .x_desc("Time (seconds)")
        .axis_desc_style(("sans-serif", 12).into_font())
        .draw()?;

    let time_range = t0..tf;
    let colors = [
        LinSrgb::new(69. / 256., 16. / 256., 129. / 256.),
        LinSrgb::new(154. / 256., 58. / 256., 127. / 256.),
        LinSrgb::new(242. / 256., 173. / 256., 78. / 256.),
        LinSrgb::new(250. / 256., 209. / 256., 152. / 256.),
    ];
    // TODO: refactor to use gradient.take(n).rev() once new version is released
    let gradient = Gradient::new(colors.iter().cloned());
    let plot_colors = gradient.take(sims.len()).collect::<Vec<Rgb<_, f64>>>();
    for (sim, color) in sims.iter().zip(plot_colors.iter().rev()) {
        let node = &sim.example_node().ok_or("no nodes")?;
        let blockchain_view: Vec<_> = node
            .tip
            .iter_blockchain(&sim.known_blocks)
            .filter(|b| time_range.contains(&b.block.timestamp))
            .collect();
        chart.draw_series(PointSeries::of_element(
            //(0..6).map(|x| ((x - 3) as f32 / 1.0, ((x - 3) as f32 / 1.0).sin())),
            blockchain_view
                .iter()
                .rev()
                .map(|&b| (b.block.timestamp as f32, b.block.weight() as f32)),
            1, // size of the point
            ShapeStyle::from(color).filled(),
            &|coord, size, style| {
                EmptyElement::at(coord) + Circle::new((0, 0), size, style)
                /*+ Text::new(
                    format!("{:?}", coord),
                    (0, 15),
                    ("sans-serif", 15).into_font(),
                )*/
            },
        ))?;
        //.label("blocks")
        //.legend(|(x, y)| Path::new(vec![(x, y), (x + 20, y)], &color));
    }

    // only draw for first simulation
    for sim in sims.iter().take(1) {
        use plotters::element::Path as PathElement;
        chart
            .draw_series(LineSeries::new(
                sim.iter_measurements()
                    .skip(1) // first one always seems to draw a vertical line
                    .map(|(t, m)| (t as f32, m.estimated_network_weight as f32)),
                &RED,
            ))?
            .label("hashrate")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        /*chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;*/
    }

    Ok(())
}

#[derive(Debug, StructOpt)]
struct ConvertOpt {
    /// Optional JSON file to load simulator states from
    #[structopt(short, long)]
    input: PathBuf,

    /// Optional JSON file to load simulator states from
    #[structopt(short, long)]
    output: PathBuf,

    /// Serialize a pretty JSON
    #[structopt(long)]
    pretty: bool,
    // TODO: filter, merge, split, etc
}

fn main_convert(opt: ConvertOpt) -> Result<(), Box<dyn Error>> {
    let sims: Vec<Simulator> = load_sims(opt.input)?;
    save_sims(opt.output, &sims, opt.pretty)?;
    Ok(())
}

fn load_sims<P: AsRef<Path>>(path: P) -> Result<Vec<Simulator>, Box<dyn Error>> {
    use std::io::BufReader;
    let path = path.as_ref();
    let sims: Vec<Simulator> = if path.extension() == Some(OsStr::new("bin")) {
        info!("deserialize Bincode from {:?}", path);
        let f = BufReader::new(File::open(path)?);
        bincode::deserialize_from(f)?
    } else {
        info!("deserialize JSON from {:?}", path);
        let f = BufReader::new(File::open(path)?);
        serde_json::from_reader(f)?
    };
    info!("deserialized, total {} sims", sims.len());
    Ok(sims)
}

fn save_sims<P: AsRef<Path>>(
    path: P,
    sims: &Vec<Simulator>,
    pretty: bool,
) -> Result<(), Box<dyn Error>> {
    use std::io::BufWriter;
    let path = path.as_ref();
    if path.extension() == Some(OsStr::new("bin")) {
        info!("serialize Bincode to {:?}, total {} sims", path, sims.len());
        let f = BufWriter::new(File::create(path)?);
        bincode::serialize_into(f, &sims)?;
    } else {
        info!("serialize JSON to {:?}", path);
        let f = BufWriter::new(File::create(path)?);
        // Serialize it to a JSON string.
        if pretty {
            serde_json::to_writer_pretty(f, &sims)?;
        } else {
            serde_json::to_writer(f, &sims)?;
        }
    }
    info!("serialized");
    Ok(())
}
