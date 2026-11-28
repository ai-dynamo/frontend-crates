"""One owner for workspace Rust targets; Cargo manifests remain the dependency authority."""

load("@crates//:crates.bzl", "aliases", "all_crate_deps")
load("@rules_rust//cargo:defs.bzl", "cargo_toml_env_vars")
load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_library", "rust_test")

def workspace_crate(
        path,
        crate_name,
        deps = [],
        dev_deps = [],
        main_only = False,
        test_compile_data = {},
        test_binaries = {},
        unit_rustc_env = {},
        rustc_flags = [],
        unit_size = "medium"):
    """Build one Cargo package at `path` as `//:<path with _>` plus its binaries and tests.

    Binaries and tests follow Cargo auto-discovery: `src/bin/*.rs` and `tests/*.rs`.

    All packages share the root BUILD file, so `path` replaces the Bazel package
    for source globs, Cargo dependency lookup, and CARGO_MANIFEST_DIR.
    """
    name = path.replace("/", "_")
    env_files = [":" + name + "_cargo_env"]
    environment = {"CARGO_MANIFEST_DIR": path}
    manifests = [path + "/Cargo.toml", "Cargo.toml"]
    cargo_toml_env_vars(name = name + "_cargo_env", src = path + "/Cargo.toml", workspace = "Cargo.toml")

    def crate_deps(**kwargs):
        return all_crate_deps(package_name = path, **kwargs)

    def crate_aliases(**kwargs):
        return aliases(package_name = path, **kwargs)

    normal = deps + crate_deps(normal = True)
    macros = crate_deps(proc_macro = True)
    test_deps = dev_deps + crate_deps(normal_dev = True)
    test_macros = crate_deps(proc_macro_dev = True)
    test_aliases = crate_aliases(normal_dev = True, proc_macro_dev = True)
    fixtures = native.glob([path + "/tests/fixtures/**", path + "/tests/data/**"], allow_empty = True)
    support = native.glob([path + "/tests/support/**", path + "/tests/common/**"], allow_empty = True)

    (rust_binary if main_only else rust_library)(
        name = name,
        crate_name = crate_name,
        crate_root = path + ("/src/main.rs" if main_only else "/src/lib.rs"),
        srcs = native.glob([path + "/src/**/*.rs"], exclude = [path + "/src/bin/**"]),
        edition = "2024",
        rustc_flags = rustc_flags,
        aliases = crate_aliases(),
        deps = normal,
        proc_macro_deps = macros,
        rustc_env = environment,
        rustc_env_files = env_files,
        compile_data = manifests,
    )
    rust_test(
        name = name + "_unit_test",
        size = unit_size,
        crate = ":" + name,
        rustc_flags = rustc_flags,
        deps = test_deps,
        proc_macro_deps = test_macros,
        aliases = test_aliases,
        compile_data = manifests + fixtures + support,
        data = fixtures,
        rustc_env = environment | unit_rustc_env,
        rustc_env_files = env_files,
    )
    for source in native.glob([path + "/src/bin/*.rs"], allow_empty = True):
        binary = source.removeprefix(path + "/src/bin/").removesuffix(".rs")
        rust_binary(
            name = binary,
            crate_root = source,
            srcs = [source],
            edition = "2024",
            deps = [":" + name] + normal,
            proc_macro_deps = macros,
            aliases = crate_aliases(),
            rustc_env = environment,
            rustc_env_files = env_files,
        )
    for source in native.glob([path + "/tests/*.rs"], allow_empty = True):
        test = source.removeprefix(path + "/tests/").removesuffix(".rs")
        children = test_binaries.get(test, [])
        rust_test(
            name = name + "_" + test,
            crate_name = test,
            crate_root = source,
            srcs = [source] + native.glob([path + "/tests/common/**/*.rs", path + "/tests/support/**/*.rs"], allow_empty = True),
            edition = "2024",
            deps = [":" + name] + normal + test_deps,
            proc_macro_deps = macros + test_macros,
            aliases = test_aliases,
            compile_data = manifests + fixtures + support + test_compile_data.get(test, []),
            data = fixtures + [":" + child for child in children],
            rustc_env = environment | {"CARGO_BIN_EXE_" + child: "$(rootpath :" + child + ")" for child in children},
            rustc_env_files = env_files,
        )
