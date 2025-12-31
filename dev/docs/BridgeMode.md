# Bridge Mode Documentation

## Overview

Bridge mode is a feature in Draw Things that enables API and SDK clients to route image generation requests through Draw Things' cloud infrastructure to self-hosted gRPC servers. It provides authenticated access to remote GPU resources while maintaining security through JWT-based authorization and request validation.

## How Bridge Mode Works

### Architecture Components

Bridge mode operates through three primary components:

1. **Client Application** (iOS/macOS app or API client)
2. **Cloud Proxy Service** (Draw Things cloud infrastructure)
3. **Self-Hosted gRPC Server** (your local GPU server running gRPCServerCLI)

### Request Flow

```
Client App → Cloud Service (Proxy) → gRPC Server → GPU Processing → Response
     ↑                                                                    ↓
     └────────────────────────── Streamed Results ──────────────────────┘
```

## Client Application Perspective

### Initiating Bridge Requests

When a client makes an image generation request through the Draw Things API or SDK:

1. **Request Preparation**: The client creates an `ImageGenerationRequest` with:
   - Generation configuration (model, steps, guidance scale, etc.)
   - Prompt and negative prompt
   - Input images/masks (if applicable)
   - Control hints (ControlNet, T2I-Adapter, etc.)

2. **Authentication Handler**: Before sending, the client calls an `authenticationHandler` with:
   ```swift
   authenticationHandler(
     fromBridge: Bool,           // Set to true for API/SDK requests
     encodedBlob: Data,           // Serialized request for checksum
     configuration: GenerationConfiguration,
     hasImage: Bool,
     shuffleCount: Int,
     cancellation: (@escaping () -> Void) -> Void
   ) -> String?  // Returns JWT bearer token
   ```

3. **JWT Token Generation**: The cloud service issues a JWT token containing:
   - `checksum`: SHA256 hash of the request blob for integrity verification
   - `stats`: User's request statistics for throttling
   - `nonce`: Unique value to prevent replay attacks
   - `fromBridge`: Boolean flag indicating API/SDK origin
   - `userId`: Unique user identifier
   - `userClass`: User tier (plus, community, background, banned, throttled)
   - `consumableType`: Optional boost type for priority processing
   - `generationId`: Unique ID for tracking this generation
   - `amount`: Number of boosts being consumed

4. **Request Transmission**: The client sends the request with:
   - Authorization header: `bearer <JWT_TOKEN>`
   - Chunked transfer enabled for large responses
   - SHA256-hashed content for deduplication

### Response Handling

The client receives streaming responses containing:

- **Progress Updates**: Real-time signposts (text encoded, sampling step X, image decoded, etc.)
- **Preview Images**: Optional intermediate results during generation
- **Remote Download Progress**: When the server downloads models from blob storage
- **Final Images**: Generated images in chunked format (4 MiB chunks for large images)
- **Scale Factor**: Upscaling multiplier applied

## Cloud Service (Proxy) Perspective

### Request Validation

When the proxy receives a request from a client:

1. **JWT Verification**:
   ```swift
   - Extracts bearer token from Authorization header
   - Decodes JWT using ES256 signature verification
   - Validates against public key from api.drawthings.ai/key
   - Checks token expiration (exp field)
   ```

2. **Request Integrity**:
   ```swift
   - Computes SHA256 checksum of request blob
   - Compares with payload.checksum
   - Validates request hasn't been tampered with
   ```

3. **Replay Attack Prevention**:
   ```swift
   - Checks if payload.nonce has been used before
   - Maintains nonce cache (configurable size limit)
   - Rejects requests with duplicate nonces
   ```

4. **Throttle Policy Enforcement**:
   ```swift
   // Standard throttle policies
   "15_min": 300,        // Max 300 requests per 15 minutes
   "10_min": 200,        // Max 200 requests per 10 minutes
   "5_min": 100,         // Max 100 requests per 5 minutes
   "1_hour": 1000,       // Max 1000 requests per hour
   "1_min": 30,          // Max 30 requests per minute
   "24_hour": 1000,      // Max 1000 requests per 24 hours (community)
   "24_hour_plus": 5000, // Max 5000 requests per 24 hours (plus users)

   // API-specific limits (when fromBridge = true)
   "24_hour_api": 100,      // API limit for community users
   "24_hour_api_plus": 500, // API limit for plus users

   // Queue management
   "daily_soft_limit_low": 500,
   "daily_soft_limit_high": 750,
   "high_free_worker_threshold": 8,
   "community_free_worker_threshold": 0,
   "throttle_queue_timeout_seconds": 3600,
   "task_loop_breakout_seconds": 30
   ```

   The proxy applies different limits based on:
   - User class (plus vs community)
   - Request origin (app vs API/SDK via `fromBridge` flag)
   - Boost consumption status

5. **Compute Unit Validation**:
   ```swift
   - Calculates request cost based on:
     - Model type and size
     - Image dimensions
     - Number of steps
     - LoRA count
     - ControlNet usage
     - img2img vs txt2img
   - Compares against user's compute unit threshold
   - Allows boost consumption to exceed normal limits
   ```

### Task Prioritization

The proxy assigns priority to each request:

```swift
enum ProxyTaskPriority {
  case real        // Boost requests - highest priority
  case high        // Plus users within normal limits
  case low         // Community users within normal limits
  case background  // Throttled/over-limit requests
}
```

Priority determination logic:
1. **Boost requests** → `.real` priority (always processed first)
2. **Plus users**:
   - Under daily_soft_limit_low → `.high`
   - Between soft limits → `.low`
   - Over daily_soft_limit_high → `.background`
3. **Community users**:
   - Under daily_soft_limit_low → `.low`
   - Over limit → `.background`
4. **Banned/throttled users** → `.background`

### Worker Queue Management

The proxy maintains multiple priority queues:

```swift
- realPriorityTasks: [WorkTask]       // Boost requests
- highPriorityTasks: [WorkTask]       // Plus users
- lowPriorityTasks: [WorkTask]        // Community users
- backgroundPriorityTasks: [WorkTask] // Throttled requests
```

Queue processing rules:
- When `availableWorkerCount < high_free_worker_threshold (8)`:
  - Only process `.real` and `.high` priority tasks
  - Throttle `.low` and `.background` tasks
- When workers available >= `community_free_worker_threshold (0)`:
  - Process all priority levels in order
- Background tasks older than `throttle_queue_timeout_seconds (1 hour)` are rejected

### Heartbeat Mechanism

To keep connections alive through Cloudflare:
- Proxy sends empty `ImageGenerationResponse` every 20 seconds
- Prevents timeout on long-running generations
- Cancelled when task completes or fails

## gRPC Server Perspective

### Server Setup

The gRPCServerCLI accepts requests in two modes:

1. **Direct Mode** (bridgeMode = false):
   - Local network discovery via mDNS/Bonjour
   - Optional shared secret authentication
   - No JWT validation
   - Used for local client connections

2. **Bridge Mode** (bridgeMode = true):
   - Receives requests from cloud proxy
   - Processes pre-validated requests
   - Trusts proxy authentication
   - Marked via `trace.fromBridge` flag

### Request Processing

When the server receives a generation request:

1. **Trace Flag**: Request contains `ImageGeneratorTrace(fromBridge: true)`

2. **LoRA Loading**: If request includes LoRAs not locally available:
   ```swift
   - ServerLoRALoader checks R2/S3 blob storage
   - Downloads missing LoRAs to secondary directory
   - Reports download progress via remoteDownload responses
   - Updates configuration with local paths
   ```

3. **Generation Pipeline**:
   ```swift
   1. Text Encoding    → currentSignpost: .textEncoded
   2. Image Encoding   → currentSignpost: .imageEncoded (if img2img)
   3. Sampling         → currentSignpost: .sampling(step)
   4. Image Decoding   → currentSignpost: .imageDecoded
   5. Face Restoration → currentSignpost: .faceRestored (if enabled)
   6. Upscaling        → currentSignpost: .imageUpscaled (if enabled)
   ```

4. **Progress Streaming**:
   ```swift
   - Server sends ImageGenerationResponse updates
   - Includes current signpost and expected signposts
   - Optional preview images during sampling
   - Proxy forwards to client unchanged
   ```

5. **Result Delivery**:
   ```swift
   - Compresses images using fpzip codec (if enabled)
   - Chunks large images into 4 MiB pieces
   - Streams chunks with .moreChunks state
   - Final chunk marked with .lastChunk state
   ```

### Server Configuration Options

```bash
# Basic setup
./gRPCServerCLI /path/to/models \
  --name "My GPU Server" \
  --address 0.0.0.0 \
  --port 7859

# With blob storage (for dynamic LoRA loading)
./gRPCServerCLI /path/to/models \
  --secondary-models-directory /path/to/downloaded \
  --blob-store-access-key "ACCESS_KEY" \
  --blob-store-secret "SECRET" \
  --blob-store-endpoint "https://bucket.r2.cloudflarestorage.com" \
  --blob-store-bucket "loras"

# With shared secret (for direct connections)
./gRPCServerCLI /path/to/models \
  --shared-secret "YOUR_SECRET"

# Performance tuning
./gRPCServerCLI /path/to/models \
  --gpu 0 \                        # GPU index
  --weights-cache 16 \             # 16 GiB weights cache
  --no-flash-attention \           # Disable for RTX 20xx series
  --cpu-offload                    # Offload weights to CPU

# Production deployment
./gRPCServerCLI /path/to/models \
  --supervised \                                    # Auto-restart on crash
  --max-crashes-within-time-window 3 \             # Max 3 crashes
  --crash-time-window 60 \                         # Within 60 seconds
  --cancellation-warning-timeout 300 \             # Warn after 5 min
  --cancellation-crash-timeout 300 \               # Crash after 5 more min
  --echo-on-queue                                  # Health check on work queue
```

### Multi-GPU Setup (Proxy Mode)

For load balancing across multiple GPUs:

```bash
# GPU Server 1 (RTX 4090)
./gRPCServerCLI /path/to/models --port 7859

# GPU Server 2 (RTX 4090)
./gRPCServerCLI /path/to/models --port 7860

# Control Server (can be on same machine)
./gRPCServerCLI /path/to/models \
  --join '{
    "host": "proxy.example.com",
    "port": 8080,
    "servers": [
      {"address": "192.168.1.10", "port": 7859, "priority": 1},
      {"address": "192.168.1.11", "port": 7860, "priority": 1}
    ]
  }'
```

Priority values:
- `1` = High priority worker (for plus users)
- `2` = Low priority worker (for community users)

## Security Model

### Authentication Chain

1. **Client → Cloud Proxy**:
   - User authenticated via Draw Things account
   - Cloud service signs JWT with private key
   - Token includes request checksum and user metadata

2. **Cloud Proxy → gRPC Server**:
   - Proxy validates JWT signature
   - Checks throttle policies and compute units
   - Only forwards validated requests

3. **gRPC Server**:
   - Trusts proxy validation (when fromBridge = true)
   - Optional shared secret for direct connections
   - No JWT validation (already done by proxy)

### Protection Mechanisms

1. **Replay Attack Prevention**:
   - Unique nonce per request
   - Nonce cache prevents reuse
   - Configurable cache size limit

2. **Request Integrity**:
   - SHA256 checksum in JWT payload
   - Validates request hasn't been modified
   - Computed on client, verified on proxy

3. **Throttle Protection**:
   - Multi-window rate limiting (1min, 5min, 15min, 1hour, 24hour)
   - User-class based limits
   - API-specific limits for bridge mode
   - Soft limits trigger priority downgrade

4. **Compute Unit Enforcement**:
   - Cost calculated before processing
   - Prevents expensive requests beyond quota
   - Boost system allows temporary overrides

## Boost System

Boosts allow users to bypass compute unit limits:

### Client Perspective

1. User purchases boosts in app
2. Client includes in JWT:
   ```swift
   consumableType: .boost
   amount: 3              // Number of boosts to spend
   generationId: "uuid"   // Unique generation ID
   ```
3. Request gets `.real` priority (highest)
4. Compute unit threshold increased by `amount * computeUnitPerBoost`

### Proxy Perspective

1. Validates boost request
2. Assigns `.real` priority (bypasses queues)
3. Tracks boost consumption
4. On success: Reports completion to billing service
5. On failure: Cancels boost, refunds user

### Server Perspective

1. Processes boost requests with highest priority
2. No special handling (same generation pipeline)
3. Boost tracking handled by proxy

## Error Handling

### Common Error Codes

```swift
.ok                   // Success
.cancelled            // User cancelled request
.permissionDenied     // Failed authentication or throttling
.dataLoss             // Checksum mismatch or corrupt data
.unimplemented        // Feature not supported
.internalError        // Server-side failure
```

### Specific Error Messages

1. **"Service bear-token is empty"**:
   - Client didn't include Authorization header
   - Fix: Ensure authentication handler returns valid JWT

2. **"Service bear-token signature is failed"**:
   - JWT checksum doesn't match request
   - Fix: Ensure request isn't modified after signing

3. **"used nonce"**:
   - Replay attack detected
   - Fix: Generate new nonce for each request

4. **"user failed to pass throttlePolicy"**:
   - Rate limit exceeded
   - Response includes: `"throttlePolicy"` in message
   - Client handler: `requestExceedLimitHandler()`

5. **"cost X exceed threshold Y"**:
   - Request too expensive for user quota
   - Fix: Reduce steps, resolution, or use boost

## Monitoring and Health Checks

### Server Health Check

```bash
# Echo call to verify server is responsive
grpcurl -plaintext -d '{"name": "health-check"}' \
  localhost:7859 \
  ImageGenerationService/Echo
```

Response includes:
- Server identifier
- Available models (if model browsing enabled)
- Shared secret status
- Compute unit thresholds

### Proxy Health Check

The proxy performs periodic health checks on workers:
- Every 10 seconds when worker available
- Removes worker after 3 consecutive failures
- Automatically re-adds when connection restored

### Echo on Queue

Use `--echo-on-queue` flag to make echo calls execute on the generation queue:
- Tests queue responsiveness
- Detects deadlocks or stalls
- Useful for monitoring systems

## Performance Tuning

### Weights Caching

```bash
--weights-cache 16  # 16 GiB cache
```

Benefits:
- Faster model switching
- Reduced VRAM thrashing
- Better multi-model performance

Trade-off: Uses more system RAM

### Response Compression

Default: Enabled

```bash
--no-response-compression  # Disable if network is fast
```

Benefits:
- Reduced bandwidth (up to 10x)
- Faster transfer over slow connections

Trade-off: CPU overhead for compression

### Flash Attention

Default: Enabled (if GPU supports it)

```bash
--no-flash-attention  # Disable for RTX 20xx series
```

Benefits:
- Faster sampling (up to 2x)
- Lower VRAM usage

Compatibility: RTX 30xx and newer

### CPU Offload

```bash
--cpu-offload  # For cards with limited VRAM
```

Offloads some weights to system RAM:
- Enables larger models on smaller GPUs
- Trade-off: Slower inference

## Best Practices

### For Server Operators

1. **Use supervision in production**:
   ```bash
   --supervised --max-crashes-within-time-window 3
   ```

2. **Configure blob storage for LoRAs**:
   - Enables dynamic model loading
   - Reduces local storage requirements

3. **Set appropriate weights cache**:
   - 8-16 GiB for single model
   - 32+ GiB for multi-model setups

4. **Monitor with health checks**:
   - Use `--echo-on-queue` for queue monitoring
   - Set up external monitoring of echo endpoint

5. **Use cancellation timeouts**:
   ```bash
   --cancellation-warning-timeout 300 \
   --cancellation-crash-timeout 300
   ```
   Prevents hung generations from blocking queue

### For API/SDK Developers

1. **Handle throttle errors gracefully**:
   - Implement exponential backoff
   - Show user-friendly quota messages
   - Suggest boost purchases

2. **Stream progress updates**:
   - Display generation progress to users
   - Show preview images if available
   - Enable cancellation

3. **Implement proper error handling**:
   - Distinguish transient vs permanent errors
   - Retry on network failures
   - Don't retry on permission errors

4. **Optimize request sizes**:
   - Compress images before upload
   - Use SHA256 deduplication
   - Enable chunked responses

## Troubleshooting

### "Permission denied" errors

Check:
1. JWT token is valid and not expired
2. Nonce is unique (not reused)
3. Checksum matches request blob
4. User hasn't exceeded throttle limits

### Slow generation

Check:
1. Server queue depth (priority level)
2. Network latency (client to proxy to server)
3. Model download in progress
4. GPU utilization on server

### Connection timeouts

Check:
1. Heartbeat mechanism working (20s interval)
2. Firewall not blocking gRPC traffic
3. Cloudflare timeout limits
4. Server responsiveness (use echo)

### Missing models

Check:
1. Model exists in server's models directory
2. LoRA loader configured for dynamic download
3. Blob storage credentials valid
4. Network connectivity to R2/S3

## Architecture Diagrams

### Standard Bridge Request Flow

```
┌─────────────┐
│ Client App  │
│ (iOS/Mac)   │
└──────┬──────┘
       │ 1. Create request
       │ 2. Get JWT from auth handler
       ▼
┌─────────────────────┐
│ Cloud Proxy Service │
│  - Validate JWT     │
│  - Check throttles  │
│  - Check compute    │
│  - Assign priority  │
└──────┬──────────────┘
       │ 3. Forward validated request
       ▼
┌──────────────────┐
│ gRPC Server      │
│  - Load models   │
│  - Generate      │
│  - Stream back   │
└──────┬───────────┘
       │ 4. Stream results
       ▼
    Proxy → Client
```

### Multi-GPU Proxy Setup

```
                    ┌──────────────┐
                    │ Cloud Proxy  │
                    └───────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ GPU Server │  │ GPU Server │  │ GPU Server │
    │  (High)    │  │  (High)    │  │  (Low)     │
    │  RTX 4090  │  │  RTX 4090  │  │  RTX 3090  │
    └────────────┘  └────────────┘  └────────────┘
```

## Future Enhancements

Potential improvements to bridge mode:

1. **Request deduplication**: Cache identical requests
2. **Smart routing**: Direct model-specific requests to appropriate GPUs
3. **Bandwidth optimization**: Adaptive compression based on network speed
4. **Priority boosting**: Dynamic priority based on wait time
5. **Fallback mechanisms**: Retry on different worker if one fails
