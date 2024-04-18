pub unsafe fn align<'a>(
    mut a: &'a [u8],
    mut a_len: usize,
    mut b: &'a [u8],
    mut b_len: usize,
    k: i32,
) -> Option<i32> {
    if a_len > b_len {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut a_len, &mut b_len);
    }

    let a_len = a_len as i32;
    let b_len = b_len as i32;
    let len_diff = b_len - a_len;

    if len_diff > k {
        return None;
    }

    let main_diag = k + 1;
    let mut curr_fr = vec![-1i32; (k as usize) * 2 + 3];
    let mut prev_fr = vec![-1i32; (k as usize) * 2 + 3];

    for curr_k in 0..=k {
        let lo = main_diag - curr_k.min((k - len_diff) / 2);
        let hi = main_diag + curr_k.min((k - len_diff) / 2 + len_diff);

        for d in lo..=hi {
            let prev_fr_ptr = prev_fr.as_ptr();
            let mut fr = (*prev_fr_ptr.add((d - 1) as usize))
                .max(*prev_fr_ptr.add(d as usize) + 1)
                .max(*prev_fr_ptr.add((d + 1) as usize) + 1);

            fr += lcp(a.as_ptr(), a_len, fr, b.as_ptr(), b_len, d - main_diag + fr);
            fr = fix_out_of_bounds(fr, d, main_diag, a_len, b_len);
            *curr_fr.as_mut_ptr().add(d as usize) = fr;
        }

        let diag_end_b_hi = hi - main_diag + a_len;

        if b_len <= diag_end_b_hi {
            let d = len_diff + main_diag;
            let fr = *curr_fr.as_ptr().add(d as usize);

            if fr >= a_len {
                return Some(curr_k);
            }
        }

        std::mem::swap(&mut curr_fr, &mut prev_fr);
    }

    None
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn simd_align<'a>(
    mut a: &'a [u8],
    mut a_len: usize,
    mut b: &'a [u8],
    mut b_len: usize,
    k: i32,
) -> Option<i32> {
    if a_len > b_len {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut a_len, &mut b_len);
    }

    let a_len = a_len as i32;
    let b_len = b_len as i32;
    let len_diff = b_len - a_len;

    if len_diff > k {
        return None;
    }

    let main_diag = k + 1;
    let mut curr_fr = vec![-1i32; (k as usize) * 2 + 3];
    let mut prev_fr = vec![-1i32; (k as usize) * 2 + 3];
    const L: i32 = 8;

    for curr_k in 0..=k {
        let lo = main_diag - curr_k.min((k - len_diff) / 2);
        let hi = main_diag + curr_k.min((k - len_diff) / 2 + len_diff);
        let mut curr_lo = lo;

        while curr_lo + L <= hi + 1 {
            let prev_fr_prev =
                _mm256_loadu_si256(prev_fr.as_ptr().add((curr_lo as usize) - 1) as _);
            let prev_fr_curr = _mm256_loadu_si256(prev_fr.as_ptr().add(curr_lo as usize) as _);
            let prev_fr_next =
                _mm256_loadu_si256(prev_fr.as_ptr().add((curr_lo as usize) + 1) as _);
            let mut fr = simd_max_fr(prev_fr_prev, prev_fr_curr, prev_fr_next);

            let fr_b = simd_b_idx(fr, curr_lo, main_diag);
            let a_13 = simd_read_13(a.as_ptr(), fr);
            let b_13 = simd_read_13(b.as_ptr(), fr_b);
            let (v_lcp, mut too_long) = simd_lcp_13(a_13, b_13);
            fr = _mm256_add_epi32(fr, v_lcp);
            fr = simd_fix_out_of_bounds(fr, curr_lo, main_diag, a_len, b_len);
            _mm256_storeu_si256(curr_fr.as_ptr().add(curr_lo as usize) as _, fr);

            while too_long > 0 {
                let d = curr_lo + (too_long.trailing_zeros() as i32);
                let mut fr = *curr_fr.as_ptr().add(d as usize);
                fr += lcp(a.as_ptr(), a_len, fr, b.as_ptr(), b_len, d - main_diag + fr);
                fr = fix_out_of_bounds(fr, d, main_diag, a_len, b_len);
                *curr_fr.as_mut_ptr().add(d as usize) = fr;

                too_long &= too_long - 1;
            }

            curr_lo += L;
        }

        for d in curr_lo..=hi {
            let prev_fr_ptr = prev_fr.as_ptr();
            let mut fr = (*prev_fr_ptr.add((d - 1) as usize))
                .max(*prev_fr_ptr.add(d as usize) + 1)
                .max(*prev_fr_ptr.add((d + 1) as usize) + 1);

            fr += lcp(a.as_ptr(), a_len, fr, b.as_ptr(), b_len, d - main_diag + fr);
            fr = fix_out_of_bounds(fr, d, main_diag, a_len, b_len);
            *curr_fr.as_mut_ptr().add(d as usize) = fr;
        }

        let diag_end_b_hi = hi - main_diag + a_len;

        if b_len <= diag_end_b_hi {
            let d = len_diff + main_diag;
            let fr = *curr_fr.as_ptr().add(d as usize);

            if fr >= a_len {
                return Some(curr_k);
            }
        }

        std::mem::swap(&mut curr_fr, &mut prev_fr);
    }

    None
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn lcp(a: *const u8, a_len: i32, a_i: i32, b: *const u8, b_len: i32, b_i: i32) -> i32 {
    const L: usize = 29;
    let a_i = a_i as usize;
    let b_i = b_i as usize;
    let mut idx = 0;

    while a_i + idx < (a_len as usize) && b_i + idx < (b_len as usize) {
        let a_v = read_29(a, a_i + idx);
        let b_v = read_29(b, b_i + idx);
        let xor = a_v ^ b_v;

        if xor > 0 {
            idx += (xor.trailing_zeros() as usize) / 2;
            return idx as i32;
        }

        idx += L;
    }

    idx as i32
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn read_29(a: *const u8, i: usize) -> u64 {
    let mut v = std::ptr::read_unaligned(a.add(i / 4) as *const u64);
    v >>= (i % 4) * 2;
    v &= (-1i64 as u64) >> 6;
    v
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn fix_out_of_bounds(fr: i32, d: i32, main_diag: i32, a_len: i32, b_len: i32) -> i32 {
    fr.min(a_len).min(b_len - d + main_diag)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_b_idx(fr: __m256i, lo: i32, main_diag: i32) -> __m256i {
    let main_diag = _mm256_set1_epi32(main_diag);
    let i = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let d = _mm256_add_epi32(_mm256_set1_epi32(lo), i);
    _mm256_add_epi32(_mm256_sub_epi32(d, main_diag), fr)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_fix_out_of_bounds(
    fr: __m256i,
    lo: i32,
    main_diag: i32,
    a_len: i32,
    b_len: i32,
) -> __m256i {
    let main_diag_b_len = _mm256_set1_epi32(main_diag + b_len);
    let a_len = _mm256_set1_epi32(a_len);
    let i = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let d = _mm256_add_epi32(_mm256_set1_epi32(lo), i);
    _mm256_min_epi32(
        fr,
        _mm256_min_epi32(a_len, _mm256_sub_epi32(main_diag_b_len, d)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_max_fr(prev: __m256i, curr: __m256i, next: __m256i) -> __m256i {
    let ones = _mm256_set1_epi32(1);
    let curr = _mm256_add_epi32(curr, ones);
    let next = _mm256_add_epi32(next, ones);
    _mm256_max_epi32(prev, _mm256_max_epi32(curr, next))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_lcp_13(a: __m256i, b: __m256i) -> (__m256i, u8) {
    let xor = _mm256_xor_si256(a, b);
    let lsb = _mm256_and_si256(xor, _mm256_sub_epi32(_mm256_set1_epi32(0), xor));
    let f = _mm256_castps_si256(_mm256_cvtepi32_ps(lsb));
    let exp = _mm256_srai_epi32(f, 23);
    let v = _mm256_sub_epi32(exp, _mm256_set1_epi32(127));
    let too_long = _mm256_movemask_ps(_mm256_castsi256_ps(v)) as u8;
    let v = _mm256_srai_epi32(v, 1);
    let lcp = _mm256_min_epu32(v, _mm256_set1_epi32(13));
    (lcp, too_long)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_read_13(a: *const u8, i: __m256i) -> __m256i {
    let mut v = _mm256_i32gather_epi32(a as _, _mm256_srli_epi32(i, 2), 1);
    v = _mm256_srlv_epi32(
        v,
        _mm256_slli_epi32(_mm256_and_si256(i, _mm256_set1_epi32(0b11)), 1),
    );
    v = _mm256_and_si256(v, _mm256_set1_epi32(((-1i32 as u32) >> 6) as i32));
    v
}

static LUT: [u8; 128] = {
    let mut l = [0u8; 128];
    l[b'A' as usize] = 0b00;
    l[b'a' as usize] = 0b00;
    l[b'C' as usize] = 0b01;
    l[b'c' as usize] = 0b01;
    l[b'G' as usize] = 0b10;
    l[b'g' as usize] = 0b10;
    l[b'T' as usize] = 0b11;
    l[b't' as usize] = 0b11;
    l
};

pub fn encode(seq: &[u8], res: &mut Vec<u8>) {
    for c in seq.chunks(4) {
        let mut curr = 0u8;

        for (i, &b) in c.iter().enumerate() {
            curr |= LUT[b as usize] << (i * 2);
        }

        res.push(curr);
    }
}

const PAD: usize = 16;

pub fn pad(res: &mut Vec<u8>) {
    res.resize(res.len() + PAD, 0u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcp() {
        unsafe {
            let a = b"ATCGATCG";
            let mut a2 = Vec::new();
            encode(a, &mut a2);
            pad(&mut a2);
            let b = b"ATCGATCC";
            let mut b2 = Vec::new();
            encode(b, &mut b2);
            pad(&mut b2);

            let res = lcp(
                a2.as_ptr(),
                a.len() as i32,
                0,
                b2.as_ptr(),
                b.len() as i32,
                0,
            );
            assert_eq!(res, 7);
        }
    }

    #[test]
    fn test_align() {
        unsafe {
            let a = b"ATCGATCG";
            let mut a2 = Vec::new();
            encode(a, &mut a2);
            pad(&mut a2);
            let b = b"ATCATCC";
            let mut b2 = Vec::new();
            encode(b, &mut b2);
            pad(&mut b2);

            let res = align(&a2, a.len(), &b2, b.len(), 0);
            assert_eq!(res, None);
            let res = align(&a2, a.len(), &b2, b.len(), 1);
            assert_eq!(res, None);
            let res = align(&a2, a.len(), &b2, b.len(), 2);
            assert_eq!(res, Some(2));
            let res = align(&a2, a.len(), &b2, b.len(), 3);
            assert_eq!(res, Some(2));
        }
    }

    #[test]
    fn test_simd_lcp() {
        #[target_feature(enable = "avx2")]
        unsafe fn inner() {
            let a = _mm256_set1_epi32(0b11_10_01_00_11_10_01_00);
            let b = _mm256_set1_epi32(0b11_11_01_00_11_10_01_00);
            let (lcp, too_long) = simd_lcp_13(a, b);
            let mut res = [0i32; 8];
            _mm256_storeu_si256(res.as_mut_ptr() as _, lcp);
            assert_eq!(res, [6i32; 8]);
            assert_eq!(too_long, 0b00000000);

            let (lcp, too_long) = simd_lcp_13(a, a);
            let mut res = [0i32; 8];
            _mm256_storeu_si256(res.as_mut_ptr() as _, lcp);
            assert_eq!(res, [13i32; 8]);
            assert_eq!(too_long, 0b11111111);
        }

        unsafe { inner() }
    }

    #[test]
    fn test_simd_align() {
        unsafe {
            let a = b"ATCGATCG";
            let mut a2 = Vec::new();
            encode(a, &mut a2);
            pad(&mut a2);
            let b = b"ATCATCC";
            let mut b2 = Vec::new();
            encode(b, &mut b2);
            pad(&mut b2);

            let res = simd_align(&a2, a.len(), &b2, b.len(), 0);
            assert_eq!(res, None);
            let res = simd_align(&a2, a.len(), &b2, b.len(), 1);
            assert_eq!(res, None);
            let res = simd_align(&a2, a.len(), &b2, b.len(), 2);
            assert_eq!(res, Some(2));
            let res = simd_align(&a2, a.len(), &b2, b.len(), 3);
            assert_eq!(res, Some(2));
        }
    }

    #[test]
    fn test_align_match() {
        unsafe {
            let a = b"AAAA".repeat(100);
            let mut a2 = Vec::new();
            encode(&a, &mut a2);
            pad(&mut a2);
            let b = b"TTTT".repeat(100);
            let mut b2 = Vec::new();
            encode(&b, &mut b2);
            pad(&mut b2);

            let res1 = align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            let res2 = simd_align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            assert_eq!(res1, res2);

            let res1 = align(&a2, a.len(), &a2, a.len(), a.len() as i32);
            let res2 = simd_align(&a2, a.len(), &a2, a.len(), a.len() as i32);
            assert_eq!(res1, res2);

            let a = b"AAAA".repeat(100);
            let mut a2 = Vec::new();
            encode(&a, &mut a2);
            pad(&mut a2);
            let b = b"TTTA".repeat(100);
            let mut b2 = Vec::new();
            encode(&b, &mut b2);
            pad(&mut b2);

            let res1 = align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            let res2 = simd_align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            assert_eq!(res1, res2);

            let a = b"ATCGATCGATCGT".repeat(100);
            let mut a2 = Vec::new();
            encode(&a, &mut a2);
            pad(&mut a2);
            let b = b"ATCGATCGATCGA".repeat(100);
            let mut b2 = Vec::new();
            encode(&b, &mut b2);
            pad(&mut b2);

            let res1 = align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            let res2 = simd_align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            assert_eq!(res1, res2);

            let a = b"ATCGATCGATCGAT".repeat(100);
            let mut a2 = Vec::new();
            encode(&a, &mut a2);
            pad(&mut a2);
            let b = b"ATCGATCGATCGAA".repeat(100);
            let mut b2 = Vec::new();
            encode(&b, &mut b2);
            pad(&mut b2);

            let res1 = align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            let res2 = simd_align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            assert_eq!(res1, res2);

            let a = b"ATGATCATCGAT".repeat(100);
            let mut a2 = Vec::new();
            encode(&a, &mut a2);
            pad(&mut a2);
            let b = b"ATCGATCGATCGAA".repeat(100);
            let mut b2 = Vec::new();
            encode(&b, &mut b2);
            pad(&mut b2);

            let res1 = align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            let res2 = simd_align(&a2, a.len(), &b2, b.len(), a.len().max(b.len()) as i32);
            assert_eq!(res1, res2);
        }
    }
}
