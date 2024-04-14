pub fn fracminhash(seq: &[u8], k: usize, frac: usize, res: &mut Vec<(u32, usize)>) {
    let threshold = std::u64::MAX / (frac as u64);
    let mut curr = 0u64;

    for (i, w) in seq.windows(k).enumerate() {
        curr |= ;
        let h = hash_u64(curr) as u32;

        if h <= frac {
            res.push((h, i));
        }
    }
}

pub fn align_u32(mut a: &[u32], mut b: &[u32], k: i32) -> i32 {
    if a.len() > b.len() {
        std::mem::swap(&mut a, &mut b);
    }

    let len_diff = (b.len() - a.len()) as i32;

    if len_diff > k {
        return None;
    }

    let main_diag = k + 1;
    let mut curr_fr = vec![-1i32; k * 2 + 3];
    let mut prev_fr = vec![-1i32; k * 2 + 3];

    for curr_k in 0..=k {
        let lo = ;
        let hi = ;

        for d in lo..hi {

        }

        std::mem::swap(&mut curr_fr, &mut prev_fr);
    }
}

pub fn align(mut a: &[u8], mut b: &[u8], k: i32) -> i32 {
    if a.len() > b.len() {
        std::mem::swap(&mut a, &mut b);
    }

    let a_len = a.len() as i32;
    let b_len = b.len() as i32;
    let len_diff = b_len - a_len;

    if len_diff > k {
        return None;
    }

    let main_diag = k + 1;
    let mut curr_fr = vec![-1i32; k * 2 + 3];
    let mut prev_fr = vec![-1i32; k * 2 + 3];

    for curr_k in 0..=k {
        let lo = main_diag - (curr_k.min((k - len_diff) / 2) + 1);
        let hi = main_diag + (curr_k.min((k - len_diff) / 2 + len_diff) + 1);

        for d in lo..hi {
            let fr = curr_fr[(d - 1) as usize]
                .max(curr_fr[d as usize] + 1)
                .max(curr_fr[(d + 1) as usize] + 1);

            if fr < a_len && d - main_diag + fr < b_len {
                fr += extend(&a[fr as usize..], &b[(d - main_diag + fr) as usize..]);
            }

            if d + a_len - main_diag == b_len && fr >= a_len {
                return Some(curr_k);
            }
        }

        std::mem::swap(&mut curr_fr, &mut prev_fr);
    }

    None
}

pub fn lcp(a: &[u8], b: &[u8]) -> i32 {
    const L: usize = 29;
    let mut idx = 0;

    while idx + L <= a.len() && idx + L <= b.len() {
        let a_v = read_29(a, idx);
        let b_v = read_29(b, idx);
        let xor = a_v ^ b_v;

        if xor > 0 {
            idx += (xor.trailing_zeros() as usize) / 2;
            return idx as i32;
        }

        idx += L;
    }

    idx // TODO
}

pub fn read_29(a: &[u8], i: usize) -> u64 {
    let mut v = std::ptr::read_unaligned(a.as_ptr().add(i / 4) as *const u64);
    v >>= (i % 4) * 2;
    v &= (-1i64 as u64) >> 6;
    v
}

pub fn simd_lcp(a: __m256i, b: __m256i) -> __m256i {
    let xor = _mm256_xor_si256(a, b);
    let lsb = _mm256_and_si256(xor, _mm256_sub_epi32(0, xor));
    let f = _mm256_castps_si256(_mm256_cvtepi32_ps(xor));
    let exp = _mm256_srai_epi32(f, 23);
    let v = _mm256_sub_epi32(exp, _mm256_set1_epi8(127));
    _mm256_srai_epi32(v, 1)
}

pub fn simd_read_13(a: &[u8], i: __m256i) -> __m256i {
    let mut v = _mm256_i32gather_epi32(a.as_ptr() as _, _mm256_srli_epi32(i, 2));
    v = _mm256_srlv_epi32(v, _mm256_slli_epi32(_mm256_and_si256(i, _mm256_set1_epi32(0b11)), 1));
    v = _mm256_and_si256(v, _mm256_set1_epi32(((-1i32 as u32) >> 6) as i32));
    v
}
