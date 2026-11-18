// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! VP8 and VP9 on the CPU through libvpx, loaded at runtime.
//!
//! Like `nvdec`, this `dlopen`s the library (`libvpx.so.9`, `.8`, `.7`, or
//! the unversioned name) and resolves each function by name, so nothing is
//! linked at build time: without libvpx, decoding returns
//! [`MmError::Unsupported`]. The decoder is created with the ABI version of
//! the headers below; a libvpx built with another one refuses it, which is
//! also reported as unsupported.
//!
//! Where libvpx comes from is the deployment's choice. `ai-dynamo/dynamo`'s
//! images build libvpx v1.14.1 from source (`container/context.yaml`,
//! `container/templates/wheel_builder.Dockerfile`) and copy `libvpx.so*` into
//! the runtime image (`container/templates/dynamo_runtime.Dockerfile`): that
//! is the library this loader finds
//! there, and the version these definitions were transcribed from. The
//! crate's CI installs a distro package (Ubuntu's `libvpx9`, 1.14.0, same
//! ABI) only to run the tests.
//!
//! The `#[repr(C)]` definitions below are transcribed from libvpx v1.14.1
//! (`vpx/vpx_codec.h`, `vpx/vpx_decoder.h`, `vpx/vpx_image.h`); sizes and
//! offsets are checked against gcc's for those headers on x86_64 (LP64).
//! libvpx carries this notice:
//!
//! > Copyright (c) 2010, The WebM Project authors. All rights reserved.
//! >
//! > Redistribution and use in source and binary forms, with or without
//! > modification, are permitted provided that the following conditions are
//! > met:
//! >
//! > * Redistributions of source code must retain the above copyright notice,
//! >   this list of conditions and the following disclaimer.
//! > * Redistributions in binary form must reproduce the above copyright
//! >   notice, this list of conditions and the following disclaimer in the
//! >   documentation and/or other materials provided with the distribution.
//! > * Neither the name of Google, nor the WebM Project, nor the names of its
//! >   contributors may be used to endorse or promote products derived from
//! >   this software without specific prior written permission.
//! >
//! > THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//! > IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
//! > THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//! > PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
//! > CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//! > EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//! > PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//! > PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//! > LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//! > NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//! > SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use std::ffi::{CStr, c_char, c_int, c_long, c_uint, c_void};
use std::sync::OnceLock;

use super::nvdec::{open, symbol};
use super::seek::{self, Order};
use super::{Matrix, Nv12, Sampler, Track, VideoCodec};
use crate::{MmError, Result};

// ---------------------------------------------------------------------------
// libvpx ABI (from libvpx v1.14.1; see the module notice)
// ---------------------------------------------------------------------------

/// `vpx_codec_err_t`.
type CodecErr = c_int;
const VPX_CODEC_OK: CodecErr = 0;
const VPX_CODEC_MEM_ERROR: CodecErr = 2;
const VPX_CODEC_ABI_MISMATCH: CodecErr = 3;
const VPX_CODEC_INCAPABLE: CodecErr = 4;
const VPX_CODEC_UNSUP_FEATURE: CodecErr = 6;
/// `VPX_DECODER_ABI_VERSION`: 3 + `VPX_CODEC_ABI_VERSION` (4 +
/// `VPX_IMAGE_ABI_VERSION`, 5).
const DECODER_ABI_VERSION: c_int = 12;
/// `VPX_IMG_FMT_I420`: 8-bit planar 4:2:0.
const IMG_FMT_I420: c_int = 0x102;

/// `vpx_codec_ctx_t`, which the caller allocates and libvpx fills in.
#[repr(C)]
struct CodecCtx {
    name: *const c_char,
    iface: *const c_void,
    err: CodecErr,
    err_detail: *const c_char,
    init_flags: c_long,
    config: *const c_void,
    private: *mut c_void,
}

/// `vpx_codec_dec_cfg_t`.
#[repr(C)]
struct DecCfg {
    threads: c_uint,
    w: c_uint,
    h: c_uint,
}

/// `vpx_image_t`.
#[repr(C)]
struct Image {
    fmt: c_int,
    cs: c_int,
    range: c_int,
    w: c_uint,
    h: c_uint,
    bit_depth: c_uint,
    d_w: c_uint,
    d_h: c_uint,
    r_w: c_uint,
    r_h: c_uint,
    x_chroma_shift: c_uint,
    y_chroma_shift: c_uint,
    planes: [*mut u8; 4],
    stride: [c_int; 4],
    bps: c_int,
    user_priv: *mut c_void,
    img_data: *mut u8,
    img_data_owner: c_int,
    self_allocd: c_int,
    fb_priv: *mut c_void,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<CodecCtx>() == 56);
    assert!(offset_of!(CodecCtx, err) == 16);
    assert!(offset_of!(CodecCtx, private) == 48);
    assert!(size_of::<DecCfg>() == 12);
    assert!(size_of::<Image>() == 136);
    assert!(offset_of!(Image, d_w) == 24);
    assert!(offset_of!(Image, planes) == 48);
    assert!(offset_of!(Image, stride) == 80);
    assert!(offset_of!(Image, user_priv) == 104);
    assert!(offset_of!(Image, fb_priv) == 128);
};

// ---------------------------------------------------------------------------
// Runtime loading
// ---------------------------------------------------------------------------

type Iface = *const c_void;

/// The libvpx entry points this module calls.
struct Api {
    vp8_dx: unsafe extern "C" fn() -> Iface,
    vp9_dx: unsafe extern "C" fn() -> Iface,
    dec_init_ver:
        unsafe extern "C" fn(*mut CodecCtx, Iface, *const DecCfg, c_long, c_int) -> CodecErr,
    decode: unsafe extern "C" fn(*mut CodecCtx, *const u8, c_uint, *mut c_void, c_long) -> CodecErr,
    get_frame: unsafe extern "C" fn(*mut CodecCtx, *mut *const c_void) -> *const Image,
    destroy: unsafe extern "C" fn(*mut CodecCtx) -> CodecErr,
    error: unsafe extern "C" fn(*const CodecCtx) -> *const c_char,
    error_detail: unsafe extern "C" fn(*const CodecCtx) -> *const c_char,
}

fn load(names: &[&CStr]) -> std::result::Result<Api, String> {
    let lib = open(names).ok_or("libvpx could not be loaded")?;
    // SAFETY: each type below matches the C declaration in libvpx's headers.
    unsafe {
        Ok(Api {
            vp8_dx: symbol(lib, c"vpx_codec_vp8_dx")?,
            vp9_dx: symbol(lib, c"vpx_codec_vp9_dx")?,
            dec_init_ver: symbol(lib, c"vpx_codec_dec_init_ver")?,
            decode: symbol(lib, c"vpx_codec_decode")?,
            get_frame: symbol(lib, c"vpx_codec_get_frame")?,
            destroy: symbol(lib, c"vpx_codec_destroy")?,
            error: symbol(lib, c"vpx_codec_error")?,
            error_detail: symbol(lib, c"vpx_codec_error_detail")?,
        })
    }
}

/// The process-wide libvpx API, loaded on first use.
fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| load(&[c"libvpx.so.9", c"libvpx.so.8", c"libvpx.so.7", c"libvpx.so"]))
        .as_ref()
        .map_err(|reason| MmError::unsupported(format!("CPU decoding is not available: {reason}")))
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// A decoder instance, destroyed when dropped. Boxed so the context stays at
/// the address libvpx was initialised with.
struct Decoder {
    api: &'static Api,
    ctx: Box<CodecCtx>,
}

impl Decoder {
    fn new(api: &'static Api, codec: VideoCodec) -> Result<Self> {
        // SAFETY: the interface getters take no arguments.
        let iface = unsafe {
            match codec {
                VideoCodec::Vp8 => (api.vp8_dx)(),
                VideoCodec::Vp9 => (api.vp9_dx)(),
                other => {
                    return Err(MmError::unsupported(format!(
                        "{} is not decoded on the CPU",
                        other.name()
                    )));
                }
            }
        };
        // SAFETY: an all-zero context (null pointers, zero integers) is what
        // vpx_codec_dec_init expects to fill in.
        let mut ctx: Box<CodecCtx> = Box::new(unsafe { std::mem::zeroed() });
        // One thread: the caller's, as with every other decode in this crate.
        let cfg = DecCfg {
            threads: 1,
            w: 0,
            h: 0,
        };
        // SAFETY: valid context, interface and config; the ABI version is the
        // one the structs above were transcribed from.
        let code = unsafe { (api.dec_init_ver)(&mut *ctx, iface, &cfg, 0, DECODER_ABI_VERSION) };
        if code != VPX_CODEC_OK {
            // A failed init leaves nothing to destroy.
            let failed = |reason: &str| {
                format!(
                    "libvpx could not create a {} decoder: {reason}",
                    codec.name()
                )
            };
            return Err(match code {
                VPX_CODEC_MEM_ERROR => MmError::internal(failed("out of memory")),
                VPX_CODEC_ABI_MISMATCH => {
                    MmError::unsupported(failed("the installed libvpx has a different ABI"))
                }
                _ => MmError::unsupported(failed(&format!("error {code}"))),
            });
        }
        Ok(Self { api, ctx })
    }

    /// libvpx's message for the last error, with its detail if any.
    fn error_text(&self) -> String {
        let text = |p: *const c_char| {
            // SAFETY: libvpx returns null or a NUL-terminated string that
            // lives at least as long as the context.
            (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        };
        // SAFETY: the context is valid, initialised or not.
        let (error, detail) = unsafe {
            (
                text((self.api.error)(&*self.ctx)),
                text((self.api.error_detail)(&*self.ctx)),
            )
        };
        match (error, detail) {
            (Some(e), Some(d)) => format!("{e} ({d})"),
            (Some(e), None) => e,
            (None, _) => "unknown error".into(),
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        if !self.ctx.private.is_null() {
            // SAFETY: an initialised context, destroyed once.
            unsafe {
                (self.api.destroy)(&mut *self.ctx);
            }
        }
    }
}

/// What to hand libvpx for `packet`, once no frame size in its headers is
/// larger than the container's `width` x `height`: libvpx has no size limit
/// of its own and allocates for a new size before decoding a frame. Sizes a
/// frame takes from a reference were checked when that reference was.
///
/// Given a VP9 packet without a superframe index, libvpx decodes frame after
/// frame until the bytes run out, and where a frame ends only shows by
/// decoding it. So such a packet is handed over with an index naming it one
/// frame, and libvpx decodes only the frame whose header was checked.
fn libvpx_input<'a>(
    codec: VideoCodec,
    packet: &'a [u8],
    width: u32,
    height: u32,
    buffer: &'a mut Vec<u8>,
) -> Result<&'a [u8]> {
    // libvpx reads an empty VP9 packet's last byte before its start.
    let Some(&first) = packet.first() else {
        return Err(MmError::invalid_input("empty video packet"));
    };
    let (sizes, as_is) = match codec {
        VideoCodec::Vp8 => {
            let size = super::vp8_keyframe_size(packet);
            // libvpx takes a keyframe too short to carry its size for an
            // inter frame, then reads the size past the packet's end.
            if size.is_none() && first & 1 == 0 {
                return Err(MmError::invalid_input(
                    "truncated or malformed VP8 keyframe header",
                ));
            }
            (size.into_iter().collect(), true)
        }
        _ => {
            let superframe = super::vp9_superframe(packet)?;
            let indexed = superframe.is_some();
            let frames = superframe.unwrap_or_else(|| vec![packet]);
            let mut sizes = Vec::new();
            for frame in frames {
                sizes.extend(super::vp9_frame_size(frame)?);
            }
            (sizes, indexed)
        }
    };
    for (w, h) in sizes {
        if w > width || h > height {
            return Err(MmError::limit_exceeded(format!(
                "{w}x{h} frames are larger than the {width}x{height} the container declares"
            )));
        }
    }
    if as_is {
        return Ok(packet);
    }
    let len = u32::try_from(packet.len())
        .map_err(|_| MmError::invalid_input("video packet too large"))?;
    // Superframe marker 0b110_11_000: one frame, its size in 4 bytes.
    const MARKER: u8 = 0xD8;
    buffer.clear();
    buffer.extend_from_slice(packet);
    buffer.push(MARKER);
    buffer.extend_from_slice(&len.to_le_bytes());
    buffer.push(MARKER);
    Ok(buffer)
}

/// Refuse an output frame the stream may not change to: the format and size
/// are checked on every frame, sampled or not, as NVDEC checks every sequence.
fn check_image(image: &Image, width: u32, height: u32) -> Result<()> {
    if image.fmt != IMG_FMT_I420 || image.bit_depth != 8 {
        return Err(MmError::unsupported("only 8-bit 4:2:0 video is supported"));
    }
    if (image.d_w, image.d_h) != (width, height) {
        return Err(MmError::invalid_input(format!(
            "{}x{} frames do not match the {width}x{height} the container declares",
            image.d_w, image.d_h
        )));
    }
    Ok(())
}

/// Decode a VP8 or VP9 `track` on the CPU and feed displayed frames to
/// `sampler`.
pub(super) fn decode(track: &Track<'_>, sampler: &mut Sampler<'_>) -> Result<()> {
    // VP8 is always BT.601 limited range; VP9 says in its keyframes.
    let (matrix, full_range) = match track.codec {
        VideoCodec::Vp9 => track.vp9_colour.get()?.unwrap_or((Matrix::Bt601, false)),
        _ => (Matrix::Bt601, false),
    };
    let mut decoder = Decoder::new(api()?, track.codec)?;
    let (mut raw, mut framed, mut uv) = (Vec::new(), Vec::new(), Vec::new());
    // Each run starts at a key frame, which libvpx decodes without the
    // references it replaces, so the skipped packets need no reset.
    let order = Order::new(track);
    let runs = seek::plan(track, &order, sampler.targets)?;
    for index in runs.iter().flat_map(|run| run.start..run.end) {
        let packet = &track.packets[index];
        if sampler.done() {
            break;
        }
        let data = libvpx_input(
            track.codec,
            track.packet_bytes(packet, &mut raw)?,
            sampler.width,
            sampler.height,
            &mut framed,
        )?;
        let len = c_uint::try_from(data.len())
            .map_err(|_| MmError::invalid_input("video packet too large"))?;
        let api = decoder.api;
        // SAFETY: `data` lives until the call returns; libvpx copies what it
        // keeps between calls.
        let code = unsafe {
            (api.decode)(
                &mut *decoder.ctx,
                data.as_ptr(),
                len,
                std::ptr::null_mut(),
                0,
            )
        };
        if code != VPX_CODEC_OK {
            let message = format!("libvpx: {}", decoder.error_text());
            // UNSUP_BITSTREAM is also what a broken frame header gets.
            return Err(match code {
                VPX_CODEC_INCAPABLE | VPX_CODEC_UNSUP_FEATURE => MmError::unsupported(message),
                VPX_CODEC_MEM_ERROR => MmError::internal(message),
                _ => MmError::invalid_input(message),
            });
        }
        // VP8 and VP9 do not reorder: a packet's frame comes out of the call
        // that decoded it, so it has that packet's timestamp.
        let mut iter: *const c_void = std::ptr::null();
        let mut outputs = 0;
        loop {
            // SAFETY: valid context and iterator; the image stays valid until
            // the next decode call.
            let image = unsafe { (api.get_frame)(&mut *decoder.ctx, &mut iter) };
            if image.is_null() {
                break;
            }
            outputs += 1;
            if outputs > 1 || !packet.displays {
                return Err(MmError::invalid_input(
                    "libvpx output a frame the packet's headers do not show",
                ));
            }
            // SAFETY: non-null, valid until the next decode call.
            let image = unsafe { &*image };
            check_image(image, sampler.width, sampler.height)?;
            let Some(shown) = order.shown[index] else {
                continue;
            };
            if !sampler.wants(shown) {
                continue;
            }
            let picture = to_nv12(image, matrix, full_range, &mut uv)?;
            sampler.push(&picture, packet.pts, shown)?;
        }
        if outputs == 0 && packet.displays {
            return Err(MmError::invalid_input(
                "libvpx output no frame for a packet whose headers show one",
            ));
        }
    }
    Ok(())
}

/// View a libvpx I420 image (see `check_image`) as NV12, interleaving its
/// chroma into `uv`.
fn to_nv12<'a>(
    image: &'a Image,
    matrix: Matrix,
    full_range: bool,
    uv: &'a mut Vec<u8>,
) -> Result<Nv12<'a>> {
    let (width, height) = (image.d_w as usize, image.d_h as usize);
    let stride = |plane: usize| {
        usize::try_from(image.stride[plane])
            .map_err(|_| MmError::internal("libvpx returned a negative stride"))
    };
    let (y_stride, u_stride, v_stride) = (stride(0)?, stride(1)?, stride(2)?);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    if width == 0 || height == 0 || y_stride < width || u_stride.min(v_stride) < chroma_width {
        return Err(MmError::internal("libvpx returned an inconsistent image"));
    }
    // SAFETY: libvpx's planes hold `stride` bytes per row for the displayed
    // rows (`d_h`, and half of it rounded up for chroma); each slice ends at
    // the last displayed byte.
    let plane = |p: usize, stride: usize, w: usize, h: usize| unsafe {
        std::slice::from_raw_parts(image.planes[p], (h - 1) * stride + w)
    };
    let y = plane(0, y_stride, width, height);
    let u = plane(1, u_stride, chroma_width, chroma_height);
    let v = plane(2, v_stride, chroma_width, chroma_height);
    uv.clear();
    uv.reserve(chroma_width * 2 * chroma_height);
    for row in 0..chroma_height {
        let (u_row, v_row) = (&u[row * u_stride..], &v[row * v_stride..]);
        for col in 0..chroma_width {
            uv.extend_from_slice(&[u_row[col], v_row[col]]);
        }
    }
    let uv: &'a Vec<u8> = uv;
    Ok(Nv12 {
        width: image.d_w,
        height: image.d_h,
        y,
        uv,
        y_stride,
        uv_stride: chroma_width * 2,
        matrix,
        full_range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_is_reported_not_fatal() {
        let err = load(&[c"libdynamo-no-such-vpx.so"]).err().unwrap();
        assert!(err.contains("could not be loaded"), "{err}");
    }

    /// The first packet of a fixture: a 256x256 VP9 keyframe in vp9.webm, one
    /// claiming 16384x16384 in vp9_huge.webm.
    fn first_packet(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/video_decode")
            .join(name);
        let bytes = std::fs::read(path).unwrap();
        let track = super::super::demux(&bytes).unwrap();
        track
            .packet_bytes(&track.packets[0], &mut Vec::new())
            .unwrap()
            .to_vec()
    }

    #[test]
    fn headers_that_would_make_libvpx_allocate_more_are_refused() {
        let input = |codec, packet: &[u8]| {
            libvpx_input(codec, packet, 256, 256, &mut Vec::new()).map(|_| ())
        };
        let limited = |r: Result<()>| matches!(r, Err(MmError::LimitExceeded { .. }));
        let invalid = |r: Result<()>| matches!(r, Err(MmError::InvalidInput { .. }));
        // VP8 keyframe header: frame tag, start code, 14-bit width / height.
        let vp8 = |w: u16, h: u16| {
            let mut key = vec![0x10, 0, 0, 0x9d, 0x01, 0x2a];
            key.extend_from_slice(&w.to_le_bytes());
            key.extend_from_slice(&h.to_le_bytes());
            key
        };
        assert!(input(VideoCodec::Vp8, &vp8(256, 256)).is_ok());
        assert!(limited(input(VideoCodec::Vp8, &vp8(512, 256))));
        assert!(invalid(input(VideoCodec::Vp8, &vp8(256, 256)[..8])));
        assert!(invalid(input(VideoCodec::Vp9, &[])));
        // A superframe whose last frame, the huge one, runs into the index:
        // libvpx decodes it, so it is checked.
        let (small, huge) = (first_packet("vp9.webm"), first_packet("vp9_huge.webm"));
        let mut packet = [&small[..], &huge].concat();
        let marker = 0b1100_1001; // two frames, 2-byte sizes
        let last = u16::try_from(huge.len() + 6).unwrap();
        packet.push(marker);
        packet.extend_from_slice(&u16::try_from(small.len()).unwrap().to_le_bytes());
        packet.extend_from_slice(&last.to_le_bytes());
        packet.push(marker);
        assert!(limited(input(VideoCodec::Vp9, &packet)));
    }

    /// A VP9 uncompressed header from (value, bit width) fields, most
    /// significant bit first, padded with frame data.
    fn vp9_header(fields: &[(u32, usize)]) -> Vec<u8> {
        let bits: Vec<u8> = fields
            .iter()
            .flat_map(|&(v, n)| (0..n).rev().map(move |i| ((v >> i) & 1) as u8))
            .collect();
        let mut out: Vec<u8> = bits
            .chunks(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0, |b, (i, bit)| b | bit << (7 - i))
            })
            .collect();
        out.extend_from_slice(&[0x55; 32]);
        out
    }

    /// Profile 0 headers up to the frame size, then the size (`w`, `h`).
    /// After frame_marker 2, profile 00 and show_existing_frame 0:
    /// frame_type 1 (inter), show_frame, error_resilient_mode 0, intra_only,
    /// reset_frame_context 0.
    fn intra_only(w: u32, h: u32) -> Vec<u8> {
        vp9_header(&[
            (2, 2),
            (0, 3),
            (1, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 2),
            (0x49_83_42, 24),
            (0xFF, 8), // refresh_frame_flags
            (w - 1, 16),
            (h - 1, 16),
        ])
    }

    fn inter(found_ref: bool, w: u32, h: u32) -> Vec<u8> {
        vp9_header(&[
            (2, 2),
            (0, 3),
            (1, 1),
            (1, 1),
            (0, 1),
            (0, 2),
            (0x01, 8), // refresh_frame_flags
            (0, 12),   // ref_frame_idx and sign bias, 3 times
            (u32::from(found_ref), 1),
            (0, 2), // found_ref for the other two
            (w - 1, 16),
            (h - 1, 16),
        ])
    }

    #[test]
    fn every_explicit_vp9_frame_size_is_read() {
        let size = |frame: &[u8]| super::super::vp9_frame_size(frame).unwrap();
        assert_eq!(size(&intra_only(200, 100)), Some((200, 100)));
        assert_eq!(size(&inter(false, 300, 50)), Some((300, 50)));
        assert_eq!(size(&inter(true, 300, 50)), None);
        let input = |frame: &[u8]| {
            libvpx_input(VideoCodec::Vp9, frame, 256, 256, &mut Vec::new()).map(|_| ())
        };
        for frame in [intra_only(16384, 16), inter(false, 16, 16384)] {
            let r = input(&frame);
            assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
        }
        assert!(input(&intra_only(256, 256)).is_ok());
    }

    #[test]
    fn a_vp9_header_cut_anywhere_before_its_size_is_refused() {
        // Every prefix of a header ends before the size it would give: none
        // may reach libvpx, which allocates as soon as it reads a size.
        let huge = first_packet("vp9_huge.webm");
        for frame in [
            &huge[..],
            &intra_only(16384, 16384),
            &inter(false, 16384, 16384),
        ] {
            let mut buffer = Vec::new();
            for len in 1..frame.len() {
                let r = libvpx_input(VideoCodec::Vp9, &frame[..len], 256, 256, &mut buffer);
                assert!(r.is_err(), "a {len}-byte prefix was handed to libvpx");
            }
        }
    }

    #[test]
    fn a_packet_without_an_index_is_decoded_as_one_frame() {
        // Without the index libvpx would go on to decode the huge keyframe
        // behind the checked one.
        let packet = [first_packet("vp9.webm"), first_packet("vp9_huge.webm")].concat();
        let mut buffer = Vec::new();
        let data = libvpx_input(VideoCodec::Vp9, &packet, 256, 256, &mut buffer).unwrap();
        let mut decoder = Decoder::new(api().unwrap(), VideoCodec::Vp9).unwrap();
        let api = decoder.api;
        // SAFETY: as in `decode`.
        let image = unsafe {
            let code = (api.decode)(
                &mut *decoder.ctx,
                data.as_ptr(),
                data.len() as c_uint,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(code, VPX_CODEC_OK, "{}", decoder.error_text());
            let mut iter: *const c_void = std::ptr::null();
            let image = (api.get_frame)(&mut *decoder.ctx, &mut iter);
            assert!(!image.is_null(), "no frame");
            &*image
        };
        assert_eq!((image.d_w, image.d_h), (256, 256));
    }
}
