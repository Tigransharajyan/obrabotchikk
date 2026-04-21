use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
use std::hint::{black_box, spin_loop};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const CACHE_LINE_BYTES: usize = 64;
const BATCH_SIZE: usize = 1024;
const UNROLL: usize = 8;
const RING_SIZE: usize = 1 << 16;
const WARMUP_MS: u64 = 500;
const BENCH_SECONDS: u64 = 3;

const PHASE_INIT: u8 = 0;
const PHASE_WARMUP: u8 = 1;
const PHASE_MEASURE: u8 = 2;
const PHASE_STOP: u8 = 3;

const MIX_A: u64 = 0x9E37_79B1_85EB_CA87;
const MIX_B: u64 = 0xC2B2_AE3D_27D4_EB4F;
const SEED_BASE: u64 = 0xD6E8_FD50_2A9C_8A5D;

#[repr(align(64))]
struct CachePadded<T>(T);

#[repr(align(64))]
struct AlignedRing {
    data: [u64; RING_SIZE],
}

struct Shared {
    phase: CachePadded<AtomicU8>,
    ready: CachePadded<AtomicUsize>,
}

struct WorkerResult {
    processed: u64,
    checksum: u64,
}

#[inline(always)]
fn mix(value: u64) -> u64 {
    let x = value.wrapping_mul(MIX_A).rotate_left(17);
    let x = x ^ (x >> 23);
    x.wrapping_mul(MIX_B) ^ (x >> 29)
}

#[inline(always)]
unsafe fn process_one(slot: *mut u64, state: &mut u64) {
    let input = *slot;
    let output = mix(input ^ *state).wrapping_add(*state);
    *slot = output;
    *state = state.rotate_left(9) ^ output.wrapping_mul(MIX_A);
}

#[inline(always)]
unsafe fn process_batch(base: *mut u64, cursor: usize, state: &mut u64) -> usize {
    let mut ptr = base.add(cursor);
    let end = ptr.add(BATCH_SIZE);

    while ptr != end {
        process_one(ptr, state);
        process_one(ptr.add(1), state);
        process_one(ptr.add(2), state);
        process_one(ptr.add(3), state);
        process_one(ptr.add(4), state);
        process_one(ptr.add(5), state);
        process_one(ptr.add(6), state);
        process_one(ptr.add(7), state);
        ptr = ptr.add(UNROLL);
    }

    let next = cursor + BATCH_SIZE;
    if next == RING_SIZE {
        0
    } else {
        next
    }
}

#[inline(always)]
fn sample_checksum(ring: &[u64; RING_SIZE], state: u64) -> u64 {
    let mut acc = state;
    let mut index = 0;
    let stride = CACHE_LINE_BYTES / std::mem::size_of::<u64>();

    while index < RING_SIZE {
        acc ^= ring[index].rotate_left((index as u32) & 31);
        index += stride;
    }

    acc
}

fn allocate_ring() -> Box<AlignedRing> {
    unsafe {
        let layout = Layout::new::<AlignedRing>();
        let ptr = alloc_zeroed(layout) as *mut AlignedRing;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Box::from_raw(ptr)
    }
}

fn init_ring(thread_id: usize) -> Box<AlignedRing> {
    let mut ring = allocate_ring();
    let data = &mut ring.data;
    let mut state = SEED_BASE ^ ((thread_id as u64 + 1).wrapping_mul(MIX_B));
    let mut index = 0;

    while index < RING_SIZE {
        state = mix(state.wrapping_add(index as u64));
        data[index] = state;
        index += 1;
    }

    ring
}

fn worker_main(shared: Arc<Shared>, thread_id: usize) -> WorkerResult {
    let mut ring = init_ring(thread_id);
    let base = ring.data.as_mut_ptr();
    let mut cursor = 0usize;
    let mut state = SEED_BASE ^ ((thread_id as u64 + 1).wrapping_mul(MIX_A));
    let mut processed = 0u64;

    shared.ready.0.fetch_add(1, Ordering::Release);

    while shared.phase.0.load(Ordering::Acquire) == PHASE_INIT {
        spin_loop();
    }

    let mut phase = shared.phase.0.load(Ordering::Relaxed);

    loop {
        let current = shared.phase.0.load(Ordering::Relaxed);
        if current != phase {
            phase = current;
            if phase == PHASE_MEASURE {
                processed = 0;
            } else if phase == PHASE_STOP {
                break;
            }
        }

        unsafe {
            cursor = process_batch(base, cursor, &mut state);
        }

        if phase == PHASE_MEASURE {
            processed += BATCH_SIZE as u64;
        }
    }

    WorkerResult {
        processed,
        checksum: sample_checksum(&ring.data, black_box(state)),
    }
}

fn main() {
    assert!(BATCH_SIZE % UNROLL == 0);
    assert!(RING_SIZE % BATCH_SIZE == 0);

    let workers = thread::available_parallelism().map_or(1, |count| count.get());
    let shared = Arc::new(Shared {
        phase: CachePadded(AtomicU8::new(PHASE_INIT)),
        ready: CachePadded(AtomicUsize::new(0)),
    });

    let mut handles = Vec::with_capacity(workers);
    for thread_id in 0..workers {
        let shared = Arc::clone(&shared);
        handles.push(thread::spawn(move || worker_main(shared, thread_id)));
    }

    while shared.ready.0.load(Ordering::Acquire) != workers {
        spin_loop();
    }

    shared.phase.0.store(PHASE_WARMUP, Ordering::Release);
    thread::sleep(Duration::from_millis(WARMUP_MS));

    let start = Instant::now();
    shared.phase.0.store(PHASE_MEASURE, Ordering::Release);
    thread::sleep(Duration::from_secs(BENCH_SECONDS));
    let elapsed = start.elapsed();
    shared.phase.0.store(PHASE_STOP, Ordering::Release);

    let mut total_processed = 0u64;
    let mut checksum = 0u64;

    for handle in handles {
        let result = handle.join().expect("worker thread panicked");
        total_processed = total_processed.wrapping_add(result.processed);
        checksum ^= result.checksum;
    }

    let throughput = total_processed as f64 / elapsed.as_secs_f64();

    println!("workers: {workers}");
    println!("ring_size_per_worker: {RING_SIZE}");
    println!("batch_size: {BATCH_SIZE}");
    println!("measurement_seconds: {:.6}", elapsed.as_secs_f64());
    println!("processed_events: {total_processed}");
    println!("events_per_second: {:.3}", throughput);
    println!("million_events_per_second: {:.3}", throughput / 1_000_000.0);
    println!("checksum: {checksum:016x}");
    println!("build: cargo run --release");
}
