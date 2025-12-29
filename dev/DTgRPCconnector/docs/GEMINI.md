# Gemini Project Analysis: Draw Things Community

## Project Overview

This project is the community-facing source code for the "Draw Things" AI image generation application. The core of this specific directory is a **gRPC server** written in Swift that exposes AI image generation capabilities to clients.

The project uses the **Bazel build system** (`bazel.build`) to manage dependencies and build targets.

### Key Components

*   **gRPC Service Definition**: The gRPC contract is defined in `Libraries/GRPC/Models/Sources/imageService/imageService.proto`. This file specifies the `ImageGenerationService` and all its associated methods (`GenerateImage`, `UploadFile`, etc.) and message types.
*   **gRPC Server Implementation**: The main server entry point is `Apps/gRPCServerCLI/gRPCServerCLI.swift`. This command-line application parses arguments, initializes the AI models, and starts the gRPC server. The core service logic is implemented in `Libraries/GRPC/Server/Sources/ImageGenerationServiceImpl.swift`.
*   **Build Configuration**: The entire project is orchestrated by Bazel `BUILD` files. The primary targets for the gRPC server are defined in `Apps/BUILD` and `Libraries/GRPC/BUILD`.

## Building and Running

The official `README.md` provides detailed instructions for building and running the project.

### 1. Initial Setup

Before building, you must run the setup script. This will install Bazel and other required dependencies.

```bash
# From the draw-things-community directory
./Scripts/install.sh
```

### 2. Building the gRPC Server

Use the Bazel `build` command to compile the `gRPCServerCLI` target. The command differs slightly between platforms.

**For macOS:**

```bash
# To build for your local architecture
bazel build Apps:gRPCServerCLI-macOS

# To build a universal binary (as done for official releases)
bazel build Apps:gRPCServerCLI-macOS --nostamp --config=release --macos_cpus=arm64,x86_64
```

**For CUDA-capable Linux:**
(Requires CUDA, Swift, and CUDNN to be installed)

```bash
# This is the command used to build the official Docker image binary
bazel build Apps:gRPCServerCLI --keep_going --spawn_strategy=local --compilation_mode=opt
```

### 3. Running the gRPC Server

Once built, the server is run as a command-line application. The main argument is the path to the directory containing the AI models.

The compiled binary will be located in a path like `bazel-bin/Apps/gRPCServerCLI-macOS`.

```bash
# Example for macOS after building
./bazel-bin/Apps/gRPCServerCLI-macOS /path/to/your/models --port 50051
```

The server accepts numerous command-line arguments for configuration, which can be inspected by running the binary with the `--help` flag.

## Development Conventions

*   **Architecture**: The project follows a client-server architecture using gRPC. The service contract is strictly defined in the `.proto` file, providing a single source of truth for API interactions.
*   **Build System**: All builds and dependencies are managed by Bazel. To add or change dependencies, you must edit the `BUILD` files.
*   **Code Generation**: The Swift code for the gRPC services and messages (e.g., in `Libraries/GRPC/Models/Sources/imageService/`) is automatically generated from the `.proto` files during the Bazel build process. You should not edit these Swift files directly. Instead, modify the `imageService.proto` file and rebuild the project.
*   **Source Code Structure**: The repository is organized into `Apps` (runnable binaries), `Libraries` (reusable code components), `Scripts` (helper scripts), and `Vendors` (third-party dependencies).
