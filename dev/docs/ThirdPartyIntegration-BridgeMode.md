# Third-Party Tool Integration: Bridge Mode Client

## Overview

This guide explains how third-party applications can integrate with the Draw Things client when it's configured to use bridge mode. This enables external tools to leverage Draw Things' cloud infrastructure and GPU resources while the client handles authentication and request routing.

## Architecture

```
Third-Party Tool → Draw Things Client (App) → Cloud Proxy → gRPC Server
                         ↑                                        ↓
                         └──────── Streamed Results ──────────────┘
```

The Draw Things client acts as a gateway, providing:
- User authentication via Draw Things account
- JWT token generation for cloud service
- Request preparation and validation
- Response streaming and decompression

## Integration Methods

### JavaScript Scripting API

Draw Things provides a JavaScript scripting environment for automating workflows and building custom integrations.

#### Environment Setup

Scripts run in a JavaScriptCore environment with access to:
- `pipeline`: Image generation and model management
- `canvas`: Canvas manipulation and layer management
- `filesystem`: File system access
- `device`: Device information
- `console`: Logging utilities

#### Basic Image Generation

```javascript
// Get current configuration
const config = pipeline.configuration;

// Modify configuration
config.model = "flux_1_schnell_f16.ckpt";
config.steps = 4;
config.width = 1024;
config.height = 1024;
config.seed = 42;
config.guidanceScale = 0.0; // Schnell doesn't use guidance

// Run generation
const result = pipeline.run({
  configuration: config,
  prompt: "A serene landscape with mountains and a lake",
  negativePrompt: ""
});

console.log("Generation completed!");
```

#### Checking Model Availability

```javascript
// Check if models are downloaded
const modelsToCheck = [
  "flux_1_schnell_f16.ckpt",
  "qwen_image_edit_2511_q8p.ckpt"
];

const downloaded = pipeline.areModelsDownloaded(modelsToCheck);
console.log("Models downloaded:", downloaded);

// Download built-in models
if (!downloaded.includes("flux_1_schnell_f16.ckpt")) {
  const success = pipeline.downloadBuiltins(["flux_1_schnell_f16.ckpt"]);
  console.log("Download initiated:", success);
}
```

#### Using LoRAs

```javascript
// Find LoRA by name
const lora = pipeline.findLoRAByName("lcm_sd_xl_base_1.0_lora_f16.ckpt");

if (lora) {
  config.loras = [{
    file: lora.file,
    weight: 1.0,
    version: lora.version
  }];

  console.log("Applied LoRA:", lora.name);
}
```

#### Using ControlNet

```javascript
// Find control by name
const control = pipeline.findControlByName("controlnet_v1.1_canny_f16.ckpt");

if (control) {
  config.controls = [{
    file: control.file,
    weight: 1.0,
    noPrompt: false,
    globalAveragePooling: false
  }];

  console.log("Applied ControlNet:", control.name);
}
```

### Advanced Features

#### Moodboard (Style Reference)

The moodboard feature allows you to provide reference images for style transfer:

```javascript
// Clear existing moodboard
canvas.clearMoodboard();

// Add images from various sources
canvas.addToMoodboardFromFiles();  // Opens file picker
canvas.addToMoodboardFromPhotos(); // Opens photo library

// Add from base64 source
const imageData = "data:image/png;base64,...";
canvas.addToMoodboardFromSrc(imageData);

// Set weight for specific image (0-2, default 1.0)
canvas.setMoodboardImageWeight(1.5, 0); // index 0, weight 1.5

// Remove image at index
canvas.removeFromMoodboardAt(1);

// Generate with moodboard active
const result = pipeline.run({
  configuration: config,
  prompt: "A portrait in the same style"
});
```

The moodboard uses the "shuffle" control hint type internally to encode style references.

#### Edit Models (Qwen Image Edit)

Edit models like Qwen Image Edit allow instruction-based image editing:

```javascript
// Load the edit model
config.model = "qwen_image_edit_2511_q8p.ckpt";

// Load base image to canvas
canvas.loadImage("/path/to/image.png");
// or from base64
canvas.loadImageSrc("data:image/png;base64,...");

// Configure for editing
config.strength = 0.75; // How much to modify (0.0-1.0)
config.steps = 28;
config.guidanceScale = 5.0;
config.sampler = SamplerType.DPMPP_2M_KARRAS;

// Run with edit instruction
const result = pipeline.run({
  configuration: config,
  prompt: "Make the sky purple and add stars",
  negativePrompt: "blurry, low quality"
});
```

Edit model variants:
- `qwen_image_edit_2511_q8p.ckpt`: 8-bit quantized (recommended)
- `qwen_image_edit_2511_q6p.ckpt`: 6-bit quantized (smaller, faster)
- `qwen_image_edit_2511_bf16_q8p.ckpt`: BF16 version (higher quality)

Configuration for edit models:
- Set `modifier: .qwenimageEdit2511` in model specification
- Supports img2img workflow with instruction-based editing
- Works best with strength 0.6-0.9

#### Mask-Based Editing

```javascript
// Load base image
canvas.loadImage("/path/to/image.png");

// Create or load mask
const mask = canvas.createMask(1024, 1024, 0); // width, height, initial value
mask.fillRectangle(100, 100, 400, 400, 255); // Fill region to inpaint

// Or use automatic masking
const foregroundMask = canvas.foregroundMask; // Auto-detect foreground
const backgroundMask = canvas.backgroundMask; // Auto-detect background

// Or body part masking
const bodyMask = canvas.bodyMask(
  [BodyMaskType.UPPER_BODY, BodyMaskType.CLOTHING],
  0.1 // extra area padding
);

// Configure for inpainting
config.strength = 1.0;
config.imagePriorSteps = 0; // Pure inpainting

const result = pipeline.run({
  configuration: config,
  prompt: "A red dress",
  mask: mask
});
```

#### Multi-Image Generation

```javascript
// Generate multiple images
config.batchCount = 4; // Number of images to generate

const result = pipeline.run({
  configuration: config,
  prompt: "A magical forest"
});

// Results are returned as array of images
```

#### Face Detection and Restoration

```javascript
// Detect faces in current canvas
const faces = canvas.detectFaces();

console.log(`Found ${faces.length} faces`);

faces.forEach((face, index) => {
  console.log(`Face ${index}:`, {
    x: face.origin.x,
    y: face.origin.y,
    width: face.size.width,
    height: face.size.height
  });

  // Move canvas to face
  canvas.moveCanvasToRect(face);
});

// Enable face restoration in config
config.faceRestoration = true;
config.faceRestorationStrength = 0.8; // 0.0-1.0
```

#### Using CLIP for Text Matching

```javascript
// Get CLIP embeddings similarity scores
const texts = [
  "a photo of a dog",
  "a photo of a cat",
  "a landscape painting"
];

const scores = canvas.CLIP(texts);
console.log("CLIP scores:", scores);

// Use to find best matching text for current image
const bestMatch = texts[scores.indexOf(Math.max(...scores))];
console.log("Best match:", bestMatch);
```

#### Visual Question Answering

```javascript
// Ask questions about the current canvas image
const answer = canvas.answer(
  "qwen3_2b_instruct_q8p.ckpt", // VQA model
  "What is the main subject of this image?"
);

console.log("Answer:", answer);
```

### File System Access

```javascript
// Access Pictures directory
const picturesPath = filesystem.pictures.path;
console.log("Pictures path:", picturesPath);

// List files in Pictures
const files = filesystem.pictures.readEntries();
files.forEach(file => {
  console.log("File:", file);
});

// List files in subdirectory
const subfolder = filesystem.pictures.readEntries("MyAlbum");

// List files in arbitrary directory
const customFiles = filesystem.readEntries("/path/to/directory");

// Save generated image
canvas.saveImage(`${picturesPath}/generated.png`, false);
// visibleRegionOnly = false saves entire canvas

// Save only visible region
canvas.saveImage(`${picturesPath}/visible.png`, true);

// Load image to canvas
canvas.loadImage(`${picturesPath}/input.png`);
```

### Working with Base64 Images

```javascript
// Load from base64
const base64Data = "data:image/png;base64,iVBORw0KG...";
canvas.loadImageSrc(base64Data);

// Save to base64
const outputBase64 = canvas.saveImageSrc(false); // full canvas
const visibleBase64 = canvas.saveImageSrc(true);  // visible only

// Use in external API
console.log("Generated image:", outputBase64);
```

### Canvas Manipulation

```javascript
// Get canvas properties
const topLeft = canvas.topLeftCorner; // {x, y}
const bounds = canvas.boundingBox;    // {x, y, width, height}
const zoom = canvas.canvasZoom;       // current zoom level

// Move canvas
canvas.moveCanvas(100, 100); // left, top offset

// Set zoom
canvas.canvasZoom = 2.0; // 2x zoom

// Update canvas size
canvas.updateCanvasSize(config);

// Clear canvas
canvas.clear();
```

### Configuration Object Reference

The `configuration` object contains all generation parameters:

```javascript
const config = {
  // Model selection
  model: "flux_1_schnell_f16.ckpt",

  // Image dimensions
  width: 1024,
  height: 1024,

  // Generation parameters
  steps: 4,
  guidanceScale: 0.0,
  seed: 42,
  strength: 1.0, // For img2img: 1.0 = full generation, 0.0 = no change

  // Sampler settings
  sampler: SamplerType.EULER_A, // See SamplerType enum

  // Batch settings
  batchCount: 1, // Number of images to generate

  // LoRAs
  loras: [
    {
      file: "lcm_sd_xl_base_1.0_lora_f16.ckpt",
      weight: 1.0,
      version: "sdxlBase"
    }
  ],

  // ControlNet
  controls: [
    {
      file: "controlnet_v1.1_canny_f16.ckpt",
      weight: 1.0,
      noPrompt: false,
      globalAveragePooling: false
    }
  ],

  // Upscaling
  upscaler: "RealESRGAN_x4plus_f16.ckpt",
  upscalerScaleFactor: 2,

  // Face restoration
  faceRestoration: false,
  faceRestorationStrength: 0.8,

  // Inpainting
  imagePriorSteps: 0, // Steps to respect original image
  maskBlur: 0.0,
  maskBlurOutset: 0,

  // Advanced
  clipSkip: 0,
  sharpness: 0.0,
  shift: 1.0, // For FLUX models

  // Mask behavior
  originalImageHeight: 0,
  originalImageWidth: 0,
  startFrameIndex: 0,
  tiledDecoding: false
};
```

### Sampler Types

```javascript
const SamplerType = {
  DPMPP_2M_KARRAS: 0,
  EULER_A: 1,
  DDIM: 2,
  PLMS: 3,
  DPMPP_SDE_KARRAS: 4,
  UNI_PC: 5,
  LCM: 6,
  EULER_A_SUBSTEP: 7,
  DPMPP_SDE_SUBSTEP: 8,
  TCD: 9,
  EULER_A_TRAILING: 10,
  DPMPP_SDE_TRAILING: 11,
  DPMPP_2M_AYS: 12,
  EULER_A_AYS: 13,
  DPMPP_SDE_AYS: 14,
  DPMPP_2M_TRAILING: 15,
  DDIM_TRAILING: 16,
  UNI_PC_TRAILING: 17,
  UNI_PC_AYS: 18
};
```

Recommended samplers by model:
- **FLUX.1 Schnell**: `EULER_A` (4 steps)
- **FLUX.1 Dev**: `EULER_A` (20-28 steps)
- **SDXL**: `DPMPP_2M_KARRAS` (20-30 steps)
- **SD v1.5**: `EULER_A` or `DPMPP_2M_KARRAS`
- **LCM models**: `LCM` sampler (4-8 steps)
- **TCD models**: `TCD` sampler (4-8 steps)

### User Input Widgets

Request information from users with custom UI:

```javascript
const response = requestFromUser(
  "Generation Settings",      // title
  "Generate",                 // confirm button text
  function() {
    // 'this' provides widget builders
    return [
      this.section("Basic", "Core settings", [
        this.textField("a beautiful sunset", "Enter prompt", true, 100),
        this.slider(7.5, this.slider.fractional(1), 0, 20, "Guidance Scale"),
        this.slider(28, null, 4, 150, "Steps")
      ]),
      this.section("Advanced", "Fine-tune generation", [
        this.switch(false, "Face Restoration"),
        this.slider(0.8, this.slider.percent, 0, 1, "Restoration Strength"),
        this.comboBox(0, ["Euler A", "DPM++ 2M", "UniPC"]),
        this.imageField("Reference Images", true) // multiSelect
      ]),
      this.section("Output", "Save options", [
        this.directory(), // Directory picker
        this.plainText("Images will be saved to selected directory"),
        this.markdown("**Note**: High resolution may take longer")
      ])
    ];
  }
);

// response is array of values in order
const [prompt, guidance, steps, faceRestore, restoreStrength,
       samplerIndex, refImages, outputDir] = response;

console.log("User selected:", {
  prompt, guidance, steps, faceRestore
});
```

Widget types:
- `size(width, height, minValue, maxValue)`: Size picker
- `slider(value, valueType, min, max, title)`: Number slider
  - `this.slider.percent`: Percentage (0-100%)
  - `this.slider.fractional(k)`: Fractional with precision k
  - `this.slider.scale`: Scale factor
- `doubleSlider([min, max], valueType, min, max, title)`: Range slider
- `textField(value, placeholder, multiline, height)`: Text input
- `imageField(title, multiSelect)`: Image picker
- `directory()`: Directory picker
- `switch(isOn, title)`: Toggle switch
- `comboBox(selectedIndex, options)`: Dropdown menu
- `segmented(index, options)`: Segmented control
- `menu(index, options)`: Menu picker
- `section(title, detail, views)`: Group widgets
- `plainText(value)`: Static text
- `markdown(value)`: Markdown text
- `image(src, height, selectable)`: Display image

### Advanced Utilities

#### Random Number Generation

```javascript
// Create RNG with seed
const rng = new RNG(42);

// Generate numbers
const num1 = rng.next();
const num2 = rng.next();

console.log("Random numbers:", num1, num2);
```

#### Geometry Classes

```javascript
// Point
const point = new Point(100, 200);

// Size
const size = new Size(800, 600);

// Rectangle
const rect = new Rectangle(10, 10, 512, 512);
console.log("Max X:", rect.maxX()); // 522
console.log("Max Y:", rect.maxY()); // 522

// Rectangle operations
const rect1 = new Rectangle(0, 0, 100, 100);
const rect2 = new Rectangle(50, 50, 100, 100);

const union = Rectangle.union(rect1, rect2);
const intersection = rect1.intersect(rect2);
const contains = rect1.contains(rect2); // false
const excluded = rect1.exclude(rect2);

rect.scale(2.0); // Scale in place
```

#### Image Metadata

```javascript
const base64Img = "data:image/png;base64,...";
const metadata = new ImageMetadata(base64Img);

console.log("Dimensions:", metadata.width, "x", metadata.height);
console.log("Size:", metadata.size.width, metadata.size.height);
```

## Practical Examples

### Batch Processing with Style Transfer

```javascript
// Get all images in a directory
const sourceDir = "MyPhotos";
const images = filesystem.pictures.readEntries(sourceDir);

// Setup moodboard style
canvas.clearMoodboard();
canvas.addToMoodboardFromFiles(); // User selects style reference
canvas.setMoodboardImageWeight(1.5, 0);

const config = pipeline.configuration;
config.model = "flux_1_dev_f16.ckpt";
config.steps = 28;
config.guidanceScale = 3.5;
config.strength = 0.7;

// Process each image
images.forEach((imagePath, index) => {
  console.log(`Processing ${index + 1}/${images.length}: ${imagePath}`);

  // Load source image
  canvas.loadImage(`${filesystem.pictures.path}/${sourceDir}/${imagePath}`);

  // Generate styled version
  const result = pipeline.run({
    configuration: config,
    prompt: "professional photography, high quality, detailed",
    negativePrompt: "blurry, low quality, distorted"
  });

  // Save result
  canvas.saveImage(
    `${filesystem.pictures.path}/Styled/${imagePath}`,
    false
  );
});

console.log("Batch processing complete!");
```

### Interactive Face Editing

```javascript
// Load portrait
canvas.loadImage("/path/to/portrait.jpg");

// Detect faces
const faces = canvas.detectFaces();

if (faces.length === 0) {
  console.error("No faces detected!");
} else {
  // Show face options to user
  const faceIndex = requestFromUser(
    "Select Face",
    "Edit",
    function() {
      return [
        this.menu(0, faces.map((f, i) => `Face ${i + 1}`))
      ];
    }
  )[0];

  // Move to selected face
  canvas.moveCanvasToRect(faces[faceIndex]);

  // Get edit instruction
  const instruction = requestFromUser(
    "Edit Face",
    "Generate",
    function() {
      return [
        this.textField("", "What changes to make?", true, 80)
      ];
    }
  )[0];

  // Setup edit model
  config.model = "qwen_image_edit_2511_q8p.ckpt";
  config.strength = 0.75;
  config.steps = 28;

  // Generate edit
  const result = pipeline.run({
    configuration: config,
    prompt: instruction,
    negativePrompt: "distorted, blurry, low quality"
  });

  canvas.saveImage("/path/to/edited_portrait.jpg", false);
}
```

### Automated Upscaling Pipeline

```javascript
const lowResImages = filesystem.pictures.readEntries("LowRes");

const config = pipeline.configuration;
config.upscaler = "RealESRGAN_x4plus_f16.ckpt";
config.upscalerScaleFactor = 4;
config.faceRestoration = true;
config.faceRestorationStrength = 0.9;

lowResImages.forEach(imagePath => {
  canvas.loadImage(`${filesystem.pictures.path}/LowRes/${imagePath}`);

  // Upscale with face restoration
  pipeline.run({
    configuration: config,
    prompt: "", // No prompt needed for upscaling
    negativePrompt: ""
  });

  // Save upscaled version
  const fileName = imagePath.replace(".jpg", "_4x.png");
  canvas.saveImage(`${filesystem.pictures.path}/Upscaled/${fileName}`, false);
});
```

### Web API Bridge

```javascript
// Simple HTTP-like interface using JavaScript
function processWebRequest(requestData) {
  const {prompt, negative_prompt, width, height, steps, seed} = requestData;

  const config = pipeline.configuration;
  config.width = width || 1024;
  config.height = height || 1024;
  config.steps = steps || 28;
  config.seed = seed || Math.floor(Math.random() * 999999);

  // Generate
  const result = pipeline.run({
    configuration: config,
    prompt: prompt,
    negativePrompt: negative_prompt || ""
  });

  // Return base64
  return canvas.saveImageSrc(false);
}

// Usage
const request = {
  prompt: "A futuristic city",
  negative_prompt: "blurry",
  width: 1024,
  height: 1024,
  steps: 28,
  seed: 12345
};

const base64Result = processWebRequest(request);
console.log("Generated:", base64Result.substring(0, 100) + "...");
```

## Best Practices

### Performance

1. **Model Caching**: Check `areModelsDownloaded()` before operations
2. **Batch Operations**: Use `batchCount` for multiple variations
3. **Canvas Reuse**: Don't clear/reload unnecessarily
4. **Progressive Enhancement**: Start with low steps, increase if needed

### Error Handling

```javascript
try {
  const downloaded = pipeline.areModelsDownloaded(["model.ckpt"]);

  if (!downloaded[0]) {
    console.warn("Model not available, downloading...");
    pipeline.downloadBuiltins(["model.ckpt"]);
    return;
  }

  const result = pipeline.run({
    configuration: config,
    prompt: "test"
  });

  if (!result) {
    console.error("Generation failed");
  }
} catch (error) {
  console.error("Error:", error);
}
```

### Resource Management

```javascript
// Clean up before heavy operations
canvas.clear();
canvas.clearMoodboard();

// Monitor device constraints
const screenSize = device.screenSize;
console.log("Screen size:", screenSize.width, "x", screenSize.height);

// Adjust based on device
if (screenSize.width < 800) {
  config.width = 512;
  config.height = 512;
} else {
  config.width = 1024;
  config.height = 1024;
}
```

## Limitations

1. **Single-threaded**: JavaScript runs on main thread
2. **No Network Access**: Cannot make external HTTP requests
3. **No File Write**: Can only save through `canvas.saveImage()`
4. **Synchronous Blocking**: `pipeline.run()` blocks until complete
5. **Bridge Mode Required**: Must use Draw Things cloud service for authentication

## Troubleshooting

### Script doesn't run
- Check JavaScript syntax
- Verify all required models are downloaded
- Check console for error messages

### Models not found
```javascript
// Always verify before use
const available = pipeline.areModelsDownloaded([modelName]);
if (!available[0]) {
  console.error(`Model ${modelName} not found`);
}
```

### Out of memory
- Reduce image dimensions
- Lower batch count
- Close other apps
- Use quantized models (q6p, q8p)

### Slow generation
- Reduce step count
- Use faster models (Schnell, LCM, TCD)
- Disable unnecessary features (face restoration, upscaling)
- Check if using bridge mode (network latency)

## Security Considerations

1. **No Direct Network**: Scripts cannot make external requests
2. **Sandboxed Environment**: Limited file system access
3. **User Authentication**: Bridge mode uses Draw Things account
4. **No Secret Exposure**: Cannot access authentication tokens
5. **Input Validation**: Validate all user inputs from `requestFromUser()`

## Next Steps

- Explore example scripts in Draw Things app
- Review model documentation for specific model requirements
- Test with bridge mode vs local generation
- Monitor usage via Draw Things dashboard
- Join Draw Things community for script sharing
