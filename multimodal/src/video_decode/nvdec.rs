// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVDEC through NVIDIA's driver, loaded at runtime.
//!
//! Like FFmpeg's `nv-codec-headers` (`dynlink_loader.h`), this `dlopen`s
//! `libcuda.so.1` and `libnvcuvid.so.1` and resolves each function by name, so
//! nothing is linked at build time and the crate builds and runs without a GPU:
//! without a driver, decoding returns [`MmError::Unsupported`].
//!
//! The `#[repr(C)]` definitions below are transcribed from `nv-codec-headers`
//! (`dynlink_cuda.h`, `dynlink_cuviddec.h`, `dynlink_nvcuvid.h`, release
//! n13.1), which carry this notice:
//!
//! > Copyright (c) 2010-2026 NVIDIA Corporation
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a
//! > copy of this software and associated documentation files (the
//! > "Software"), to deal in the Software without restriction, including
//! > without limitation the rights to use, copy, modify, merge, publish,
//! > distribute, sublicense, and/or sell copies of the Software, and to permit
//! > persons to whom the Software is furnished to do so, subject to the
//! > following conditions:
//! >
//! > The above copyright notice and this permission notice shall be included
//! > in all copies or substantial portions of the Software.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//! > OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//! > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//! > NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//! > DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//! > OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//! > USE OR OTHER DEALINGS IN THE SOFTWARE.
//!
//! Only what decoding needs is declared. Structures the parser hands to the
//! decoder (`CUVIDPICPARAMS`) are passed through as opaque pointers.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use super::{Matrix, Nv12, Sampler, Track, VideoCodec, check_size};
use crate::{MmError, Result};

// ---------------------------------------------------------------------------
// Driver ABI (from nv-codec-headers; see the module notice)
// ---------------------------------------------------------------------------

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuDevicePtr = u64;
type CuVideoDecoder = *mut c_void;
type CuVideoParser = *mut c_void;
/// `tcu_ulong` is `unsigned long`: 64 bits on the LP64 targets this builds for.
type TcuUlong = u64;

const CUDA_SUCCESS: CuResult = 0;

// cudaVideoCodec
const CODEC_H264: c_uint = 4;
const CODEC_HEVC: c_uint = 8;
const CODEC_VP9: c_uint = 10;
const CODEC_AV1: c_uint = 11;
/// cudaVideoChromaFormat_420
const CHROMA_420: c_uint = 1;
/// cudaVideoSurfaceFormat_NV12
const SURFACE_NV12: c_uint = 0;
/// cudaVideoDeinterlaceMode_Weave: no deinterlacing.
const DEINTERLACE_WEAVE: c_uint = 0;
/// cudaVideoCreate_PreferCUVID: the dedicated video engines.
const CREATE_PREFER_CUVID: TcuUlong = 0x04;
// CUvideopacketflags
const PKT_ENDOFSTREAM: TcuUlong = 0x01;
const PKT_TIMESTAMP: TcuUlong = 0x02;

/// `CUVIDDECODECAPS`.
#[repr(C)]
struct DecodeCaps {
    codec_type: c_uint,
    chroma_format: c_uint,
    bit_depth_minus8: c_uint,
    reserved1: [c_uint; 3],
    is_supported: u8,
    num_nvdecs: u8,
    output_format_mask: u16,
    max_width: c_uint,
    max_height: c_uint,
    max_mb_count: c_uint,
    min_width: u16,
    min_height: u16,
    is_histogram_supported: u8,
    counter_bit_depth: u8,
    max_histogram_bins: u16,
    is_decode_stats_supported: u8,
    reserved4: [u8; 3],
    reserved3: [c_uint; 9],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Rect32 {
    left: c_int,
    top: c_int,
    right: c_int,
    bottom: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Rect16 {
    left: i16,
    top: i16,
    right: i16,
    bottom: i16,
}

/// `CUVIDEOFORMAT`, as the parser reports a sequence.
#[repr(C)]
struct VideoFormat {
    codec: c_uint,
    frame_rate: [c_uint; 2],
    progressive_sequence: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    min_num_decode_surfaces: u8,
    coded_width: c_uint,
    coded_height: c_uint,
    display_area: Rect32,
    chroma_format: c_uint,
    bitrate: c_uint,
    display_aspect_ratio: [c_int; 2],
    /// Bit fields: `video_format:3`, `video_full_range_flag:1`, then
    /// `color_primaries`, `transfer_characteristics`, `matrix_coefficients`.
    video_signal_description: [u8; 4],
    seqhdr_data_length: c_uint,
}

/// `CUVIDDECODECREATEINFO`.
#[repr(C)]
struct DecodeCreateInfo {
    width: TcuUlong,
    height: TcuUlong,
    num_decode_surfaces: TcuUlong,
    codec_type: c_uint,
    chroma_format: c_uint,
    creation_flags: TcuUlong,
    bit_depth_minus8: TcuUlong,
    intra_decode_only: TcuUlong,
    max_width: TcuUlong,
    max_height: TcuUlong,
    reserved1: TcuUlong,
    display_area: Rect16,
    output_format: c_uint,
    deinterlace_mode: c_uint,
    target_width: TcuUlong,
    target_height: TcuUlong,
    num_output_surfaces: TcuUlong,
    vid_lock: *mut c_void,
    target_rect: Rect16,
    enable_histogram: TcuUlong,
    enable_decode_features: TcuUlong,
    reserved2: [TcuUlong; 3],
}

type SequenceCallback = unsafe extern "C" fn(*mut c_void, *mut VideoFormat) -> c_int;
type DecodeCallback = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type DisplayCallback = unsafe extern "C" fn(*mut c_void, *mut ParserDispInfo) -> c_int;

/// `CUVIDPARSERPARAMS`.
#[repr(C)]
struct ParserParams {
    codec_type: c_uint,
    max_num_decode_surfaces: c_uint,
    clock_rate: c_uint,
    error_threshold: c_uint,
    max_display_delay: c_uint,
    /// Bit fields: `bAnnexb:1` (AV1 Annex B input), `bMemoryOptimize:1`.
    flags: c_uint,
    reserved1: [c_uint; 4],
    user_data: *mut c_void,
    sequence_callback: Option<SequenceCallback>,
    decode_picture: Option<DecodeCallback>,
    display_picture: Option<DisplayCallback>,
    get_operating_point: *mut c_void,
    get_sei_msg: *mut c_void,
    reserved2: [*mut c_void; 5],
    ext_video_info: *mut c_void,
}

/// `CUVIDSOURCEDATAPACKET`.
#[repr(C)]
struct SourceDataPacket {
    flags: TcuUlong,
    payload_size: TcuUlong,
    payload: *const u8,
    timestamp: i64,
}

/// `CUVIDPARSERDISPINFO`.
#[repr(C)]
struct ParserDispInfo {
    picture_index: c_int,
    progressive_frame: c_int,
    top_field_first: c_int,
    repeat_first_field: c_int,
    timestamp: i64,
}

/// `CUVIDPROCPARAMS`.
#[repr(C)]
struct ProcParams {
    progressive_frame: c_int,
    second_field: c_int,
    top_field_first: c_int,
    unpaired_field: c_int,
    reserved_flags: c_uint,
    reserved_zero: c_uint,
    raw_input_dptr: u64,
    raw_input_pitch: c_uint,
    raw_input_format: c_uint,
    raw_output_dptr: u64,
    raw_output_pitch: c_uint,
    reserved1: c_uint,
    output_stream: *mut c_void,
    reserved: [c_uint; 46],
    histogram_dptr: *mut u64,
    proc_ext: *mut c_void,
}

// The structs above must keep the C layout exactly; a mismatch corrupts what
// the driver reads or writes. Sizes and offsets as gcc reports them for the
// n13.1 headers on x86_64 (LP64).
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<DecodeCaps>() == 88);
    assert!(size_of::<VideoFormat>() == 64);
    assert!(offset_of!(VideoFormat, video_signal_description) == 56);
    assert!(offset_of!(VideoFormat, seqhdr_data_length) == 60);
    assert!(size_of::<DecodeCreateInfo>() == 176);
    assert!(offset_of!(DecodeCreateInfo, display_area) == 80);
    assert!(offset_of!(DecodeCreateInfo, output_format) == 88);
    assert!(offset_of!(DecodeCreateInfo, vid_lock) == 120);
    assert!(offset_of!(DecodeCreateInfo, target_rect) == 128);
    assert!(size_of::<ParserParams>() == 136);
    assert!(offset_of!(ParserParams, user_data) == 40);
    assert!(offset_of!(ParserParams, sequence_callback) == 48);
    assert!(offset_of!(ParserParams, ext_video_info) == 128);
    assert!(size_of::<SourceDataPacket>() == 32);
    assert!(size_of::<ParserDispInfo>() == 24);
    assert!(size_of::<ProcParams>() == 264);
    assert!(offset_of!(ProcParams, output_stream) == 56);
    assert!(offset_of!(ProcParams, histogram_dptr) == 248);
};

// ---------------------------------------------------------------------------
// Runtime loading
// ---------------------------------------------------------------------------

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

/// The driver entry points this module calls, and GPU 0's primary context.
struct Api {
    /// GPU 0's primary context, retained once and kept for the life of the
    /// process (retaining and releasing per call would rebuild the context
    /// each time). Stored as an address so `Api` stays `Send + Sync`.
    context: usize,
    cu_ctx_push_current: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_ctx_pop_current: unsafe extern "C" fn(*mut CuContext) -> CuResult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    get_decoder_caps: unsafe extern "C" fn(*mut DecodeCaps) -> CuResult,
    create_decoder: unsafe extern "C" fn(*mut CuVideoDecoder, *mut DecodeCreateInfo) -> CuResult,
    destroy_decoder: unsafe extern "C" fn(CuVideoDecoder) -> CuResult,
    decode_picture: unsafe extern "C" fn(CuVideoDecoder, *mut c_void) -> CuResult,
    map_frame: unsafe extern "C" fn(
        CuVideoDecoder,
        c_int,
        *mut u64,
        *mut c_uint,
        *mut ProcParams,
    ) -> CuResult,
    unmap_frame: unsafe extern "C" fn(CuVideoDecoder, u64) -> CuResult,
    create_parser: unsafe extern "C" fn(*mut CuVideoParser, *mut ParserParams) -> CuResult,
    parse: unsafe extern "C" fn(CuVideoParser, *mut SourceDataPacket) -> CuResult,
    destroy_parser: unsafe extern "C" fn(CuVideoParser) -> CuResult,
}

/// Open the first library in `names` that loads.
fn open(names: &[&CStr]) -> Option<*mut c_void> {
    names.iter().find_map(|name| {
        // SAFETY: `name` is a NUL-terminated string; dlopen has no other
        // preconditions. The handle is never closed (the API lives as long
        // as the process, as in FFmpeg's loader).
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        (!handle.is_null()).then_some(handle)
    })
}

/// Resolve `symbol` in `handle` as a function of type `F`.
///
/// # Safety
///
/// `F` must be the function-pointer type of `symbol`'s C declaration.
unsafe fn symbol<F: Copy>(handle: *mut c_void, symbol: &CStr) -> std::result::Result<F, String> {
    // SAFETY: `handle` came from dlopen; `symbol` is NUL-terminated.
    let address = unsafe { dlsym(handle, symbol.as_ptr()) };
    if address.is_null() {
        return Err(format!("{} not found", symbol.to_string_lossy()));
    }
    const { assert!(std::mem::size_of::<F>() == std::mem::size_of::<*mut c_void>()) };
    // SAFETY: `F` is pointer-sized (checked above) and the caller names the
    // symbol's real signature.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&address) })
}

/// Load the driver libraries in `cuda` and `nvcuvid` and initialise CUDA.
fn load(cuda: &[&CStr], nvcuvid: &[&CStr]) -> std::result::Result<Api, String> {
    let cuda = open(cuda).ok_or("libcuda.so.1 (the NVIDIA driver) could not be loaded")?;
    let nvcuvid = open(nvcuvid).ok_or("libnvcuvid.so.1 (NVDEC) could not be loaded")?;
    // SAFETY: each type below matches the C declaration in nv-codec-headers.
    unsafe {
        let cu_init: unsafe extern "C" fn(c_uint) -> CuResult = symbol(cuda, c"cuInit")?;
        let cu_device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult =
            symbol(cuda, c"cuDeviceGet")?;
        let cu_primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult =
            symbol(cuda, c"cuDevicePrimaryCtxRetain")?;
        let mut api = Api {
            context: 0,
            cu_ctx_push_current: symbol(cuda, c"cuCtxPushCurrent_v2")?,
            cu_ctx_pop_current: symbol(cuda, c"cuCtxPopCurrent_v2")?,
            cu_memcpy_dtoh: symbol(cuda, c"cuMemcpyDtoH_v2")?,
            get_decoder_caps: symbol(nvcuvid, c"cuvidGetDecoderCaps")?,
            create_decoder: symbol(nvcuvid, c"cuvidCreateDecoder")?,
            destroy_decoder: symbol(nvcuvid, c"cuvidDestroyDecoder")?,
            decode_picture: symbol(nvcuvid, c"cuvidDecodePicture")?,
            map_frame: symbol(nvcuvid, c"cuvidMapVideoFrame64")?,
            unmap_frame: symbol(nvcuvid, c"cuvidUnmapVideoFrame64")?,
            create_parser: symbol(nvcuvid, c"cuvidCreateVideoParser")?,
            parse: symbol(nvcuvid, c"cuvidParseVideoData")?,
            destroy_parser: symbol(nvcuvid, c"cuvidDestroyVideoParser")?,
        };
        if cu_init(0) != CUDA_SUCCESS {
            return Err("cuInit failed: no usable NVIDIA GPU".into());
        }
        let (mut device, mut context): (CuDevice, CuContext) = (0, std::ptr::null_mut());
        if cu_device_get(&mut device, 0) != CUDA_SUCCESS
            || cu_primary_ctx_retain(&mut context, device) != CUDA_SUCCESS
        {
            return Err("GPU 0's CUDA context could not be retained".into());
        }
        api.context = context as usize;
        Ok(api)
    }
}

/// The process-wide driver API, loaded on first use.
fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        load(
            &[c"libcuda.so.1", c"libcuda.so"],
            &[c"libnvcuvid.so.1", c"libnvcuvid.so"],
        )
    })
    .as_ref()
    .map_err(|reason| MmError::unsupported(format!("NVDEC is not available: {reason}")))
}

fn check(code: CuResult, what: &str) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(MmError::internal(format!(
            "{what} failed (CUDA error {code})"
        )))
    }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// GPU 0's primary context, current on this thread while the guard lives.
struct Context {
    api: &'static Api,
}

impl Context {
    fn new(api: &'static Api) -> Result<Self> {
        // SAFETY: the context was retained in `load` and is never released.
        check(
            unsafe { (api.cu_ctx_push_current)(api.context as CuContext) },
            "cuCtxPushCurrent",
        )?;
        Ok(Self { api })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let mut ctx: CuContext = std::ptr::null_mut();
        // SAFETY: pops the context `new` pushed on this thread.
        unsafe {
            (self.api.cu_ctx_pop_current)(&mut ctx);
        }
    }
}

fn codec_type(codec: VideoCodec) -> Result<c_uint> {
    match codec {
        VideoCodec::H264 => Ok(CODEC_H264),
        VideoCodec::Hevc => Ok(CODEC_HEVC),
        VideoCodec::Vp9 => Ok(CODEC_VP9),
        VideoCodec::Av1 => Ok(CODEC_AV1),
        other => Err(MmError::unsupported(format!(
            "{} is not decoded by this crate",
            other.name()
        ))),
    }
}

/// What this GPU's NVDEC can decode for `codec` in 8-bit 4:2:0.
fn caps(api: &Api, codec: c_uint) -> Result<DecodeCaps> {
    // SAFETY: an all-zero `DecodeCaps` is valid (integers only).
    let mut caps: DecodeCaps = unsafe { std::mem::zeroed() };
    caps.codec_type = codec;
    caps.chroma_format = CHROMA_420;
    caps.bit_depth_minus8 = 0;
    // SAFETY: valid in/out struct; a CUDA context is current.
    check(
        unsafe { (api.get_decoder_caps)(&mut caps) },
        "cuvidGetDecoderCaps",
    )?;
    Ok(caps)
}

/// Refuse a coded size the decoder cannot handle.
fn check_caps(caps: &DecodeCaps, codec: VideoCodec, width: u32, height: u32) -> Result<()> {
    if caps.is_supported == 0 {
        return Err(MmError::unsupported(format!(
            "this GPU's NVDEC does not decode 8-bit 4:2:0 {}",
            codec.name()
        )));
    }
    let mbs = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
    if width < u32::from(caps.min_width)
        || height < u32::from(caps.min_height)
        || width > caps.max_width
        || height > caps.max_height
        || mbs > u64::from(caps.max_mb_count)
    {
        return Err(MmError::unsupported(format!(
            "{width}x{height} {} is outside this GPU's NVDEC range ({}x{} to {}x{})",
            codec.name(),
            caps.min_width,
            caps.min_height,
            caps.max_width,
            caps.max_height
        )));
    }
    Ok(())
}

/// The YUV-to-RGB matrix for ITU-T H.273 `matrix_coefficients`, which NVDEC
/// reports for H.264, HEVC and AV1.
fn h273_matrix(codec: VideoCodec, code: u8) -> Result<Matrix> {
    match code {
        1 => Ok(Matrix::Bt709),
        2 | 5 | 6 => Ok(Matrix::Bt601),
        _ => Err(MmError::unsupported(format!(
            "{} matrix coefficients {code} are not supported",
            codec.name()
        ))),
    }
}

/// What the sequence header said, once the decoder exists.
struct Surface {
    coded: (u32, u32),
    /// The display area: left, top, right, bottom.
    crop: (c_int, c_int, c_int, c_int),
    /// The displayed size, which the host copy is cropped to.
    width: u32,
    height: u32,
    /// The decoder's output height: the displayed height rounded up to even,
    /// as NVDEC requires for 4:2:0. The chroma plane starts this many rows
    /// into the mapped surface.
    target_height: u32,
    /// Decode surfaces the decoder was created with.
    surfaces: c_uint,
    matrix: Matrix,
    full_range: bool,
}

/// Shared with the parser callbacks through `user_data`.
struct State<'s, 'sp, 't, 'p> {
    api: &'static Api,
    track: &'t Track<'p>,
    /// Indices of the packets that display a frame, in presentation order;
    /// see `decode`.
    order: Vec<usize>,
    /// Frames displayed so far.
    displayed: usize,
    /// VP9 colour from the first keyframe header (NVDEC does not report it).
    vp9_colour: Option<(Matrix, bool)>,
    codec: VideoCodec,
    codec_type: c_uint,
    sampler: &'s mut Sampler<'sp>,
    decoder: CuVideoDecoder,
    surface: Option<Surface>,
    host_y: Vec<u8>,
    host_uv: Vec<u8>,
    error: Option<MmError>,
}

impl Drop for State<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if !self.decoder.is_null() {
            // SAFETY: created by cuvidCreateDecoder, destroyed once; the
            // parser that used it is gone (it is dropped first).
            unsafe {
                (self.api.destroy_decoder)(self.decoder);
            }
        }
    }
}

/// Destroys the parser when dropped.
struct Parser {
    api: &'static Api,
    handle: CuVideoParser,
}

impl Drop for Parser {
    fn drop(&mut self) {
        // SAFETY: created by cuvidCreateVideoParser, destroyed once.
        unsafe {
            (self.api.destroy_parser)(self.handle);
        }
    }
}

/// Owns the boxed `State` the parser's `user_data` points at. Every access,
/// from the callbacks and from `decode`, goes through this one raw pointer, so
/// no `&mut State` outlives a single use (the parser holds the same pointer).
struct StateBox<'s, 'sp, 't, 'p>(*mut State<'s, 'sp, 't, 'p>);

impl Drop for StateBox<'_, '_, '_, '_> {
    fn drop(&mut self) {
        // SAFETY: from `Box::into_raw` in `decode`, freed once, after the
        // parser (declared later, so dropped earlier) is destroyed.
        drop(unsafe { Box::from_raw(self.0) });
    }
}

/// Run `body` for a parser callback: errors and panics are recorded in the
/// state and reported to the parser as 0 (stop), never unwound into C. Once
/// one is recorded, later callbacks return 0 without running.
fn callback(
    user: *mut c_void,
    body: impl FnOnce(&mut State<'_, '_, '_, '_>) -> Result<c_int>,
) -> c_int {
    // SAFETY: `user` is the `StateBox` pointer passed in
    // `ParserParams::user_data`. The parser only calls back from inside
    // `cuvidParseVideoData`, while `decode` holds no reference into the state,
    // so this is the only live reference for the duration of the callback.
    let state = unsafe { &mut *user.cast::<State<'_, '_, '_, '_>>() };
    if state.error.is_some() {
        return 0;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(state))) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => {
            state.error.get_or_insert(e);
            0
        }
        Err(_) => {
            state
                .error
                .get_or_insert(MmError::internal("panic in an NVDEC callback"));
            0
        }
    }
}

unsafe extern "C" fn on_sequence(user: *mut c_void, format: *mut VideoFormat) -> c_int {
    callback(user, |state| {
        // SAFETY: the parser passes a valid format for this call.
        let format = unsafe { &*format };
        state.sequence(format)
    })
}

unsafe extern "C" fn on_decode(user: *mut c_void, picture: *mut c_void) -> c_int {
    callback(user, |state| {
        if state.decoder.is_null() {
            return Ok(0);
        }
        // SAFETY: decoder created in the sequence callback; `picture` is the
        // parser's CUVIDPICPARAMS for this call.
        check(
            unsafe { (state.api.decode_picture)(state.decoder, picture) },
            "cuvidDecodePicture",
        )?;
        Ok(1)
    })
}

unsafe extern "C" fn on_display(user: *mut c_void, info: *mut ParserDispInfo) -> c_int {
    callback(user, |state| {
        if info.is_null() {
            return Ok(1); // end of stream
        }
        if state.decoder.is_null() {
            return Ok(0);
        }
        // SAFETY: non-null, valid for this call.
        let info = unsafe { &*info };
        state.display(info)?;
        Ok(1)
    })
}

impl State<'_, '_, '_, '_> {
    fn sequence(&mut self, format: &VideoFormat) -> Result<c_int> {
        if format.chroma_format != CHROMA_420
            || format.bit_depth_luma_minus8 != 0
            || format.bit_depth_chroma_minus8 != 0
        {
            return Err(MmError::unsupported("only 8-bit 4:2:0 video is supported"));
        }
        let area = format.display_area;
        let size = |v: c_int| u32::try_from(v).ok().filter(|v| *v > 0);
        let (width, height) = size(area.right - area.left)
            .zip(size(area.bottom - area.top))
            .ok_or_else(|| MmError::invalid_input("the video's display area is empty"))?;
        let coded = (format.coded_width, format.coded_height);
        // Before the decoder allocates: a frame must fit the size the
        // container declared (coded sizes round up to the codec's block size)
        // and the caller's limits.
        let room = |v: u32| v.div_ceil(64) * 64;
        check_size(coded.0, coded.1, &self.sampler.limits)?;
        if width > self.sampler.width
            || height > self.sampler.height
            || coded.0 > room(self.sampler.width)
            || coded.1 > room(self.sampler.height)
        {
            return Err(MmError::limit_exceeded(format!(
                "{}x{} frames are larger than the {}x{} the container declares",
                coded.0, coded.1, self.sampler.width, self.sampler.height
            )));
        }
        if (width, height) != (self.sampler.width, self.sampler.height) {
            return Err(MmError::invalid_input(format!(
                "{width}x{height} frames do not match the {}x{} the container declares",
                self.sampler.width, self.sampler.height
            )));
        }
        // NVDEC scales an odd-sized AV1 frame to its even target instead of
        // decoding it one row or column larger, so the crop below would
        // return a resampled picture.
        if self.codec == VideoCodec::Av1 && (width | height | coded.0 | coded.1) & 1 == 1 {
            return Err(MmError::unsupported(
                "odd-sized AV1 frames are not supported",
            ));
        }
        check_caps(
            &caps(self.api, self.codec_type)?,
            self.codec,
            coded.0,
            coded.1,
        )?;
        // NVDEC reports the H.273 colour description for H.264, HEVC and AV1,
        // but nothing for VP9 (all zeros even for a BT.709 stream), so VP9's
        // colour comes from its keyframe header instead.
        let signal = format.video_signal_description;
        let (matrix, full_range) = match self.codec {
            VideoCodec::Vp9 => self.vp9_colour.unwrap_or((Matrix::Bt601, false)),
            codec => (h273_matrix(codec, signal[3])?, signal[0] & 0x08 != 0),
        };
        let crop = (area.left, area.top, area.right, area.bottom);
        let surfaces = c_uint::from(format.min_num_decode_surfaces.max(1));
        if let Some(surface) = &self.surface {
            // The decoder keeps the size, crop and colour it was created
            // with; a later sequence header may only repeat them.
            if surface.coded != coded
                || surface.crop != crop
                || (surface.matrix, surface.full_range) != (matrix, full_range)
            {
                return Err(MmError::invalid_input(
                    "frame size, crop or colour changed mid-stream",
                ));
            }
            // Nor can it grow the surfaces it was created with.
            if surfaces > surface.surfaces {
                return Err(MmError::unsupported(
                    "a later sequence header needs more decode surfaces",
                ));
            }
            return Ok(surface.surfaces as c_int);
        }
        // NVDEC's 4:2:0 output needs an even width and height (an odd target
        // makes it resample the picture). Round the display area, and the
        // decoder size with it when the coded size itself is odd (VP9 and AV1
        // report the exact frame size), up to even; the surfaces NVDEC
        // allocates are aligned well beyond that. The extra row or column is
        // cropped on the host.
        let even = |v: u32| v + (v & 1);
        let (target_width, target_height) = (even(width), even(height));
        let decoder_size = (even(coded.0), even(coded.1));
        if u64::from(area.left.max(0) as u32) + u64::from(target_width) > u64::from(decoder_size.0)
            || u64::from(area.top.max(0) as u32) + u64::from(target_height)
                > u64::from(decoder_size.1)
        {
            return Err(MmError::unsupported(
                "the display area cannot be rounded to an even size inside the frame",
            ));
        }
        let edge = |v: i64| {
            i16::try_from(v).map_err(|_| MmError::invalid_input("display area out of range"))
        };
        let display_area = Rect16 {
            left: edge(i64::from(area.left))?,
            top: edge(i64::from(area.top))?,
            right: edge(i64::from(area.left) + i64::from(target_width))?,
            bottom: edge(i64::from(area.top) + i64::from(target_height))?,
        };
        let mut info = DecodeCreateInfo {
            width: TcuUlong::from(decoder_size.0),
            height: TcuUlong::from(decoder_size.1),
            num_decode_surfaces: TcuUlong::from(surfaces),
            codec_type: self.codec_type,
            chroma_format: CHROMA_420,
            creation_flags: CREATE_PREFER_CUVID,
            bit_depth_minus8: 0,
            intra_decode_only: 0,
            max_width: TcuUlong::from(decoder_size.0),
            max_height: TcuUlong::from(decoder_size.1),
            reserved1: 0,
            display_area,
            output_format: SURFACE_NV12,
            deinterlace_mode: DEINTERLACE_WEAVE,
            target_width: TcuUlong::from(target_width),
            target_height: TcuUlong::from(target_height),
            num_output_surfaces: 1,
            vid_lock: std::ptr::null_mut(),
            target_rect: Rect16::default(),
            enable_histogram: 0,
            enable_decode_features: 0,
            reserved2: [0; 3],
        };
        // SAFETY: valid out-pointer and create info; a context is current.
        check(
            unsafe { (self.api.create_decoder)(&mut self.decoder, &mut info) },
            "cuvidCreateDecoder",
        )?;
        self.surface = Some(Surface {
            coded,
            crop,
            width,
            height,
            target_height,
            surfaces,
            matrix,
            full_range,
        });
        Ok(surfaces as c_int)
    }

    fn display(&mut self, info: &ParserDispInfo) -> Result<()> {
        // H.264 and HEVC frames carry their packet's presentation rank as the
        // timestamp. VP9 and AV1 never reorder, and NVDEC's timestamps slip
        // by one around VP9 superframes, so their frames are matched to the
        // displaying packets in decode order.
        let index = match self.codec {
            VideoCodec::H264 | VideoCodec::Hevc => usize::try_from(info.timestamp).ok(),
            _ => Some(self.displayed),
        };
        self.displayed += 1;
        let packet = index
            .and_then(|i| self.order.get(i))
            .map(|&i| &self.track.packets[i])
            .ok_or_else(|| MmError::internal("NVDEC displayed a frame from an unknown packet"))?;
        if !packet.shown {
            return Ok(());
        }
        // Count frames that are not sampled without copying them off the GPU.
        if !self.sampler.wants_next() {
            self.sampler.skip();
            return Ok(());
        }
        let (width, height, target_height, matrix, full_range) = self
            .surface
            .as_ref()
            .map(|s| (s.width, s.height, s.target_height, s.matrix, s.full_range))
            .ok_or_else(|| MmError::internal("NVDEC displayed a frame before its sequence"))?;
        // SAFETY: an all-zero ProcParams is valid (integers and null pointers).
        let mut params: ProcParams = unsafe { std::mem::zeroed() };
        params.progressive_frame = info.progressive_frame;
        params.top_field_first = info.top_field_first;
        params.unpaired_field = c_int::from(info.repeat_first_field < 0);
        let (mut frame, mut pitch) = (0u64, 0 as c_uint);
        // SAFETY: decoder and picture index come from the parser; the
        // out-pointers are valid.
        check(
            unsafe {
                (self.api.map_frame)(
                    self.decoder,
                    info.picture_index,
                    &mut frame,
                    &mut pitch,
                    &mut params,
                )
            },
            "cuvidMapVideoFrame",
        )?;
        let copied = self.copy_nv12(frame, pitch as usize, width, height, target_height);
        // SAFETY: unmaps exactly the frame mapped above.
        let unmapped = unsafe { (self.api.unmap_frame)(self.decoder, frame) };
        copied?;
        check(unmapped, "cuvidUnmapVideoFrame")?;
        let picture = Nv12 {
            width,
            height,
            y: &self.host_y,
            uv: &self.host_uv,
            y_stride: pitch as usize,
            uv_stride: pitch as usize,
            matrix,
            full_range,
        };
        self.sampler.push(&picture, packet.pts)
    }

    /// Copy the displayed part of a mapped NV12 surface to the host buffers.
    /// The surface holds `target_height` luma rows, then the interleaved
    /// chroma rows, all `pitch` bytes apart.
    fn copy_nv12(
        &mut self,
        frame: u64,
        pitch: usize,
        width: u32,
        height: u32,
        target_height: u32,
    ) -> Result<()> {
        let rows = height as usize;
        if pitch < width as usize {
            return Err(MmError::internal(
                "NVDEC returned a pitch narrower than the frame",
            ));
        }
        let luma = pitch * rows;
        let chroma = pitch * rows.div_ceil(2);
        let chroma_start = pitch as u64 * u64::from(target_height);
        self.host_y.resize(luma, 0);
        self.host_uv.resize(chroma, 0);
        // SAFETY: the mapped surface holds `pitch * target_height` luma bytes
        // (`target_height >= height`) followed by `pitch * target_height / 2`
        // chroma bytes (`>= pitch * ceil(height / 2)`); the host buffers are
        // exactly the copied sizes. A context is current.
        unsafe {
            check(
                (self.api.cu_memcpy_dtoh)(self.host_y.as_mut_ptr().cast(), frame, luma),
                "cuMemcpyDtoH (luma)",
            )?;
            check(
                (self.api.cu_memcpy_dtoh)(
                    self.host_uv.as_mut_ptr().cast(),
                    frame + chroma_start,
                    chroma,
                ),
                "cuMemcpyDtoH (chroma)",
            )?;
        }
        Ok(())
    }
}

/// Decode `track` on GPU 0 and feed displayed frames to `sampler`.
pub(super) fn decode(track: &Track<'_>, sampler: &mut Sampler<'_>) -> Result<()> {
    let codec_type = codec_type(track.codec)?;
    let api = api()?;
    let _context = Context::new(api)?;
    // Refuse early, before parsing, when the container size is out of range.
    check_caps(
        &caps(api, codec_type)?,
        track.codec,
        sampler.width,
        sampler.height,
    )?;
    let (prefix, length_size) = super::bitstream_prefix(track)?;

    // NVDEC treats packet timestamps as presentation times: it hands them back
    // sorted, in display order, whatever order the packets arrive in. Give
    // each packet its presentation rank, so a displayed H.264 / HEVC frame's
    // timestamp is the rank of the packet it came from (B-frames included).
    // Only packets that display a frame carry a timestamp: one that displays
    // nothing (a hidden VP9 frame) must not take a place in the sorted order.
    // VP9 and AV1 display in decode order, which `order` keeps.
    let mut order: Vec<usize> = (0..track.packets.len())
        .filter(|&i| track.packets[i].displays)
        .collect();
    if matches!(track.codec, VideoCodec::H264 | VideoCodec::Hevc) {
        order.sort_by(|&a, &b| track.packets[a].pts.total_cmp(&track.packets[b].pts));
    }
    let mut rank = vec![0usize; track.packets.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
    }

    // One colour for the whole stream: the decoder is created with it.
    let mut vp9_colour = None;
    if track.codec == VideoCodec::Vp9 {
        for packet in &track.packets {
            if let Some(colour) = super::vp9_keyframe_colour(&packet.data).transpose()? {
                if vp9_colour.is_some_and(|c| c != colour) {
                    return Err(MmError::invalid_input("VP9 colour changed mid-stream"));
                }
                vp9_colour = Some(colour);
            }
        }
    }

    let state = StateBox(Box::into_raw(Box::new(State {
        api,
        track,
        order,
        displayed: 0,
        vp9_colour,
        codec: track.codec,
        codec_type,
        sampler,
        decoder: std::ptr::null_mut(),
        surface: None,
        host_y: Vec::new(),
        host_uv: Vec::new(),
        error: None,
    })));
    let mut params = ParserParams {
        codec_type,
        max_num_decode_surfaces: 1,
        clock_rate: 0,
        error_threshold: 0,
        max_display_delay: 0,
        flags: 0,
        reserved1: [0; 4],
        user_data: state.0.cast(),
        sequence_callback: Some(on_sequence),
        decode_picture: Some(on_decode),
        display_picture: Some(on_display),
        get_operating_point: std::ptr::null_mut(),
        get_sei_msg: std::ptr::null_mut(),
        reserved2: [std::ptr::null_mut(); 5],
        ext_video_info: std::ptr::null_mut(),
    };
    let mut handle: CuVideoParser = std::ptr::null_mut();
    // SAFETY: valid params; the state outlives the parser (`parser` is
    // declared after `state`, so it is dropped first).
    check(
        unsafe { (api.create_parser)(&mut handle, &mut params) },
        "cuvidCreateVideoParser",
    )?;
    let parser = Parser { api, handle };

    let mut buffer = Vec::new();
    for (index, packet) in track.packets.iter().enumerate() {
        // SAFETY: no callback runs between parse calls; a short-lived read
        // through the one pointer.
        if unsafe { (*state.0).sampler.done() } {
            break;
        }
        let payload =
            super::to_decoder_input(packet, &prefix, length_size, index == 0, &mut buffer)?;
        let mut data = SourceDataPacket {
            flags: if packet.displays { PKT_TIMESTAMP } else { 0 },
            payload_size: payload.len() as TcuUlong,
            payload: payload.as_ptr(),
            timestamp: rank[index] as i64,
        };
        // SAFETY: `payload` lives until the call returns; the parser copies
        // what it keeps. Callbacks run inside this call.
        let code = unsafe { (api.parse)(parser.handle, &mut data) };
        // SAFETY: as above.
        if let Some(e) = unsafe { (*state.0).error.take() } {
            return Err(e);
        }
        check(code, "cuvidParseVideoData")?;
    }
    // SAFETY: as above.
    if !unsafe { (*state.0).sampler.done() } {
        let mut end = SourceDataPacket {
            flags: PKT_ENDOFSTREAM,
            payload_size: 0,
            payload: std::ptr::null(),
            timestamp: 0,
        };
        // SAFETY: an empty end-of-stream packet flushes the display queue.
        let code = unsafe { (api.parse)(parser.handle, &mut end) };
        // SAFETY: as above.
        if let Some(e) = unsafe { (*state.0).error.take() } {
            return Err(e);
        }
        check(code, "cuvidParseVideoData (end of stream)")?;
    }
    drop(parser);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_driver_libraries_are_reported_not_fatal() {
        let err = load(
            &[c"libdynamo-no-such-cuda.so"],
            &[c"libdynamo-no-such-nvcuvid.so"],
        )
        .err()
        .unwrap();
        assert!(err.contains("could not be loaded"), "{err}");
    }

    #[test]
    fn vp8_and_unknown_codecs_are_unsupported() {
        assert!(matches!(
            codec_type(VideoCodec::Vp8),
            Err(MmError::Unsupported { .. })
        ));
        assert_eq!(codec_type(VideoCodec::H264).unwrap(), CODEC_H264);
        assert_eq!(codec_type(VideoCodec::Av1).unwrap(), CODEC_AV1);
    }

    #[test]
    fn h273_matrix_codes() {
        assert_eq!(h273_matrix(VideoCodec::H264, 1).unwrap(), Matrix::Bt709);
        assert_eq!(h273_matrix(VideoCodec::Hevc, 2).unwrap(), Matrix::Bt601);
        assert_eq!(h273_matrix(VideoCodec::Av1, 6).unwrap(), Matrix::Bt601);
        assert!(h273_matrix(VideoCodec::H264, 0).is_err(), "identity (RGB)");
        assert!(h273_matrix(VideoCodec::Av1, 9).is_err(), "BT.2020");
    }

    #[test]
    fn caps_reject_sizes_outside_the_decoders_range() {
        // SAFETY: integers only.
        let mut caps: DecodeCaps = unsafe { std::mem::zeroed() };
        caps.is_supported = 1;
        caps.min_width = 48;
        caps.min_height = 16;
        caps.max_width = 4096;
        caps.max_height = 4096;
        caps.max_mb_count = 65536;
        assert!(check_caps(&caps, VideoCodec::H264, 1920, 1080).is_ok());
        assert!(check_caps(&caps, VideoCodec::H264, 32, 32).is_err());
        assert!(check_caps(&caps, VideoCodec::H264, 8192, 64).is_err());
        caps.is_supported = 0;
        assert!(check_caps(&caps, VideoCodec::H264, 1920, 1080).is_err());
    }
}
