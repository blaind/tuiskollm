//! Qualification for Qwen3.8-Flash-Next engram staging.

#[cfg(test)]
mod tests {
    use crate::device_benchmark;
    use tuisko_engine::{
        QWEN38_FLASH_NEXT_ENGRAM_WIDTHS, Qwen38FlashNextEngramStager,
        gather_qwen38_flash_next_engram_window,
    };
    use tuisko_gpu::{
        ArenaLayout, CudaContext, CudaGraph, DeviceArena, GpuError, device_memory_info,
    };
    use tuisko_model::{
        QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN, QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN,
        Qwen38FlashNextEngramCarry, Qwen38FlashNextEngramHashConstants, Qwen38FlashNextEngramTable,
    };

    type Suite = Result<(), Box<dyn std::error::Error>>;

    const HEADS: usize = 16;
    const HEAD_DIM: usize = 160;
    const SHARD_ROWS: usize = 128;
    const SHARDS: usize = 3;
    const TOKEN_BYTES: usize = HEADS * HEAD_DIM;
    const VOCABS: [i64; HEADS] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

    fn constants() -> Qwen38FlashNextEngramHashConstants {
        let constants = Qwen38FlashNextEngramHashConstants::for_qualification(VOCABS).unwrap();
        assert_eq!(constants.padded_rows(), SHARDS * SHARD_ROWS);
        constants
    }

    fn sentinel(shard: usize, byte: usize) -> u8 {
        ((shard * 0x1f + byte) % 251) as u8
    }

    fn shards() -> Vec<Vec<u8>> {
        (0..SHARDS)
            .map(|shard| {
                (0..SHARD_ROWS * HEAD_DIM)
                    .map(|byte| sentinel(shard, byte))
                    .collect()
            })
            .collect()
    }

    fn borrowed(shards: &[Vec<u8>]) -> Vec<&[u8]> {
        shards.iter().map(Vec::as_slice).collect()
    }

    fn table<'a>(shards: &'a [&'a [u8]]) -> Qwen38FlashNextEngramTable<'a> {
        Qwen38FlashNextEngramTable::new(shards, SHARD_ROWS, HEAD_DIM, constants()).unwrap()
    }

    fn tokens(seed: u64, len: usize, eos_every: usize) -> Vec<u32> {
        let mut state = seed;
        (0..len)
            .map(|index| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if eos_every > 0 && index % eos_every == eos_every - 1 {
                    QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN
                } else {
                    ((state >> 33) % 248_319) as u32
                }
            })
            .collect()
    }

    /// Literal hash law with no production carry or row-hasher calls.
    fn literal_rows(
        constants: Qwen38FlashNextEngramHashConstants,
        tokens: &[u32],
    ) -> (Vec<i64>, [u32; 2]) {
        let mut previous = [QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN; 2];
        let mut rows = Vec::with_capacity(tokens.len() * HEADS);

        for &token in tokens {
            let mut shifts = [QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN; 3];
            shifts[0] = token;
            for position in 1..3 {
                shifts[position] = previous[position - 1];
                if previous[position - 1] == QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN {
                    break;
                }
            }

            let terms = [0, 1, 2].map(|position| {
                i64::from(shifts[position])
                    .checked_mul(constants.layer_multipliers()[position])
                    .unwrap()
            });
            let hashes = [
                terms[0],
                terms[0] ^ terms[1],
                terms[0] ^ terms[1] ^ terms[2],
            ];

            for head in 0..HEADS {
                let order = 1 + head / 8;
                rows.push(
                    hashes[order] % constants.head_vocab_sizes()[head]
                        + constants.head_offsets()[head],
                );
            }

            previous = [token, previous[0]];
        }

        (rows, previous)
    }

    fn expected_bytes(rows: &[i64]) -> Vec<u8> {
        let mut expected = Vec::with_capacity(rows.len() * HEAD_DIM);
        for &row in rows {
            let row = row as usize;
            let shard = row / SHARD_ROWS;
            let shard_row = row % SHARD_ROWS;
            expected.extend((0..HEAD_DIM).map(|byte| sentinel(shard, shard_row * HEAD_DIM + byte)));
        }
        expected
    }

    fn gather(
        table: Qwen38FlashNextEngramTable<'_>,
        carry: &mut Qwen38FlashNextEngramCarry,
        tokens: &[u32],
    ) -> Vec<u8> {
        let mut rows = vec![0; tokens.len() * QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN];
        let mut destination = vec![0; tokens.len() * TOKEN_BYTES];
        gather_qwen38_flash_next_engram_window(table, carry, tokens, &mut rows, &mut destination)
            .unwrap();
        destination
    }

    #[test]
    fn qwen38_flash_next_engram_staging_suite_host_gather_matches_literal_rows() {
        let owned = shards();
        let borrowed = borrowed(&owned);
        let table = table(&borrowed);
        let window = tokens(7, 128, 11);
        let (rows, previous) = literal_rows(table.constants(), &window);
        let mut carry = Qwen38FlashNextEngramCarry::start();
        let staged = gather(table, &mut carry, &window);

        assert_eq!(staged, expected_bytes(&rows));
        assert_eq!(carry.previous(), previous);

        let mut streamed = Vec::new();
        let mut streamed_carry = Qwen38FlashNextEngramCarry::start();
        for tile in window[..64].chunks(32) {
            streamed.extend(gather(table, &mut streamed_carry, tile));
        }
        for token in &window[64..] {
            streamed.extend(gather(table, &mut streamed_carry, &[*token]));
        }

        assert_eq!(streamed, staged);
        assert_eq!(streamed_carry, carry);
    }

    #[test]
    fn qwen38_flash_next_engram_staging_suite_refusal_is_transactional() {
        let owned = shards();
        let borrowed = borrowed(&owned);
        let table = table(&borrowed);
        let mut carry = Qwen38FlashNextEngramCarry::start();
        let mut rows = vec![0x55; 32 * HEADS];
        let mut destination = vec![0xa5; 32 * TOKEN_BYTES];
        let saved_carry = carry;
        let saved_destination = destination.clone();
        let mut invalid = tokens(13, 32, 0);
        invalid[17] = 248_320;

        let error = gather_qwen38_flash_next_engram_window(
            table,
            &mut carry,
            &invalid,
            &mut rows,
            &mut destination,
        )
        .unwrap_err();

        assert!(error.to_string().contains("outside vocabulary"), "{error}");
        assert_eq!(carry, saved_carry);
        assert_eq!(destination, saved_destination);

        for width in [0, 5, 31, 33, 1_025] {
            assert!(!QWEN38_FLASH_NEXT_ENGRAM_WIDTHS.contains(&width));
        }
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn qwen38_flash_next_engram_staging_suite_device_plane_matches_eager_and_graph_consumers()
    -> Suite {
        let _preflight = device_benchmark::preflight()?;
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
            return Err("engram staging qualification requires compute capability 12.0".into());
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let owned = shards();
        let borrowed = borrowed(&owned);
        let table = table(&borrowed);
        let window = tokens(23, 64, 9);
        let mut carry = Qwen38FlashNextEngramCarry::start();
        let mut stager = Qwen38FlashNextEngramStager::new(&context)?;
        let stable_address = stager.plane_address()?;
        let (window_rows, _) = literal_rows(table.constants(), &window);

        stager.stage(&stream, table, &mut carry, &window[..32])?;
        let first = expected_bytes(&window_rows[..32 * HEADS]);
        assert_eq!(stager.read_plane(&stream, 32)?, first);

        let mut layout = ArenaLayout::new();
        let observed_region = layout.reserve::<u8>(32 * TOKEN_BYTES, 256)?;
        let observed = DeviceArena::zeroed(&stream, &layout)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            // SAFETY: both arenas outlive every replay and name disjoint allocations.
            unsafe { stager.copy_plane_to_arena_async(&stream, &observed, observed_region, 32) }
                .unwrap();
            Ok(())
        })?;

        // SAFETY: the stager and observation arena outlive this replay.
        unsafe { graph.launch(&stream)? };
        assert_eq!(observed.copy_to_host(&stream, observed_region)?, first);

        let waits = stager.reuse_waits();
        stager.stage(&stream, table, &mut carry, &window[32..])?;
        assert_eq!(stager.reuse_waits(), waits + 1);
        assert_eq!(stager.plane_address()?, stable_address);
        let second = expected_bytes(&window_rows[32 * HEADS..]);
        assert_eq!(stager.read_plane(&stream, 32)?, second);
        // SAFETY: the captured addresses remain live and stable.
        unsafe { graph.launch(&stream)? };
        assert_eq!(observed.copy_to_host(&stream, observed_region)?, second);

        let saved_carry = carry;
        let saved_staged = stager.staged_window()?.to_vec();
        let waits = stager.reuse_waits();
        let mut invalid = tokens(31, 32, 0);
        invalid[5] = 248_320;
        assert!(stager.stage(&stream, table, &mut carry, &invalid).is_err());
        assert_eq!(carry, saved_carry);
        assert_eq!(stager.reuse_waits(), waits);
        assert_eq!(stager.staged_window()?, saved_staged);

        let before = device_memory_info(&context)?;
        for _ in 0..8 {
            stager.stage(&stream, table, &mut carry, &window[..32])?;
        }
        stream.synchronize().map_err(GpuError::from)?;
        let after = device_memory_info(&context)?;
        assert_eq!(before, after, "staging allocated after warmup");
        device_benchmark::require_current_process_exclusive()?;

        Ok(())
    }
}
