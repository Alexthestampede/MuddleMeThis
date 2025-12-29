# Next Session Plan - Debug Log Capture

## Goal
Capture complete debug output from a successful generation with the official Draw Things client to understand what configuration it actually uses.

---

## Step-by-Step Instructions

### 1. Prepare to Capture Logs

On the machine running the Docker server:

```bash
# Stop current container
docker stop $(docker ps -q)

# Start container in detached mode with debug enabled
docker run -d \
  -v /dt/models:/grpc-models \
  -p 7859:7859 \
  --gpus all \
  drawthingsai/draw-things-grpc-server-cli:latest \
  gRPCServerCLI /grpc-models --model-browser --cpu-offload --supervised --debug

# Get container ID
CONTAINER_ID=$(docker ps -q)
echo "Container ID: $CONTAINER_ID"

# Start capturing logs to file
docker logs -f $CONTAINER_ID > /tmp/grpc-full-debug.log 2>&1 &
LOG_PID=$!
echo "Log capture PID: $LOG_PID"
```

### 2. Run Test Generation

Using the **official Draw Things client** (iOS/Mac app):
- Connect to server at `192.168.2.150:7859`
- Model: `realdream_15sd15_q6p_q8p.ckpt`
- Prompt: `a basket full of kittens`
- Size: **512×512**
- Steps: 16
- CFG Scale: 5.0
- Scheduler: UniPC (or whatever default)
- **Wait for generation to complete** (~10 seconds)

### 3. Stop Log Capture

```bash
# Stop the log capture process
kill $LOG_PID

# Stop the container
docker stop $CONTAINER_ID

# Verify log file exists and has content
ls -lh /tmp/grpc-full-debug.log
wc -l /tmp/grpc-full-debug.log
```

### 4. Copy Log File

```bash
# Copy to home directory for easy access
cp /tmp/grpc-full-debug.log ~/grpc-successful-512x512.log

# If needed, transfer to development machine
# scp user@server:/tmp/grpc-full-debug.log ./debug/
```

---

## What to Look For in the Log

### Key Sections:

1. **Early UNet Phase** - Should see attention operations:
   ```
   CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD
   [cnnp_reshape_build] dim: (2, ???, 8, 40)
   ```
   - The `???` should be 4,096 (64×64) for SD 1.5
   - NOT 262,144 (512×512)

2. **Model Build Phase** - Shows tensor dimensions:
   ```
   [cnnp_functional_model_build]
   [cnnp_dense_build]
   [cnnp_reshape_build]
   ```

3. **Configuration Values** - Might show what was received:
   ```
   Look for any lines mentioning:
   - start_width / start_height
   - dimensions
   - latent
   - configuration
   ```

4. **Final Output** - Should show:
   ```
   [1x4x64x64]  (latent output - CORRECT for 512×512)
   [1x512x512x3] (final RGB - CORRECT)
   ```

### Search Commands:

```bash
# Find attention operations
grep -n "ATTENTION" grpc-successful-512x512.log

# Find reshape operations with dimensions
grep -n "reshape_build.*dim:" grpc-successful-512x512.log

# Find the problem dimensions
grep -n "262144\|1048576\|4096\|16384" grpc-successful-512x512.log

# Find configuration-related lines
grep -n -i "config\|width\|height\|dimension" grpc-successful-512x512.log
```

---

## Alternative: Docker Logs Afterward

If real-time capture doesn't work, you can get logs after the fact:

```bash
# Run generation first, then:
docker logs $(docker ps -q) > /tmp/grpc-debug.log 2>&1

# Or with timestamps:
docker logs -t $(docker ps -q) > /tmp/grpc-debug-timestamped.log 2>&1
```

---

## Questions to Answer

From the captured log, we need to determine:

1. **What tensor dimensions are actually used?**
   - Does it show 4,096 tokens (correct) or 262,144 (wrong)?

2. **What happens in the early UNet phase?**
   - We only have the VAE decoder phase from successdebug.txt
   - Need to see the attention operations

3. **Is the latent output correct?**
   - Should be `[1x4x64x64]` not `[1x4x512x512]`

4. **Are there any configuration hints?**
   - Does the log show what values were received?
   - Any fields we're not setting?

---

## After Capture

Share the log file and we can:
1. Compare with our failing configurations
2. Identify the critical difference
3. Update our client to match
4. Test and verify it works

---

## Your Ideas

*(Space for you to add your ideas for investigation)*

-
-
-

---

*Ready to continue when you are!*
