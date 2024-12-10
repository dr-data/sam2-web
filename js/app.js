import SAM2Predictor from './decoder.js';
import SAM2Encoder from './encoder.js';

/**
 * Global variables
 * These store state and references needed across various functions.
 */
let embedding = null;
let encoder = null;
let predictor = null;
let originalImage = null;
let imageWidth = 0;
let imageHeight = 0;
let points = [];
let isNegative = false;
let isLoading = false;

/**
 * Constants
 * Test images that can be quickly loaded.
 */
const TEST_IMAGE_URLS = {
    image1: 'https://storage.googleapis.com/lb-artifacts-testing-public/sam2/plants.jpg',
    image2: 'https://storage.googleapis.com/lb-artifacts-testing-public/sam2/truck.jpg'
};

/**
 * Draw a segmentation mask onto the mask canvas.
 * @param {HTMLCanvasElement} maskCanvas - The canvas on which to draw the mask.
 * @param {Object} maskData - The mask data returned by the predictor.
 * @param {number} imageWidth - The width of the image.
 * @param {number} imageHeight - The height of the image.
 */
function drawMask(maskCanvas, maskData, imageWidth, imageHeight) {
    const maskCtx = maskCanvas.getContext('2d');
    const width = maskData.dims[3];
    const height = maskData.dims[2];

    // Convert the mask tensor data into image data
    const maskImageData = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
        const value = maskData.data[i];
        const pixelIndex = i * 4;

        if (value > 0.0) {
            // Mark pixels inside the mask as semi-transparent red
            maskImageData[pixelIndex] = 255;     // R
            maskImageData[pixelIndex + 1] = 0;   // G
            maskImageData[pixelIndex + 2] = 0;   // B
            maskImageData[pixelIndex + 3] = 128; // A (transparency)
        } else {
            // Pixels outside the mask are transparent
            maskImageData[pixelIndex + 3] = 0;
        }
    }

    const imageData = new ImageData(maskImageData, width, height);

    // Use a temporary canvas to resize the mask
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(imageData, 0, 0);

    // Clear previous mask and draw the new mask at the correct size
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.drawImage(tempCanvas, 0, 0, imageWidth, imageHeight);
}

/**
 * Draw a point (positive or negative) on the source canvas.
 * @param {CanvasRenderingContext2D} ctx - The drawing context.
 * @param {Object} point - The point object with x, y, and type properties.
 */
function drawPoint(ctx, point) {
    ctx.fillStyle = (point.type === 1) ? 'green' : 'red';
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
}

/**
 * Load an image from a given URL.
 * @param {string} url - The URL of the image.
 * @returns {Promise<HTMLImageElement>} A promise that resolves with the loaded image.
 */
async function loadImageFromURL(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous'; // Enable CORS for external images
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load image from ${url}`));
        img.src = url;
    });
}

/**
 * Update the loading state of the application (show/hide spinner).
 * @param {boolean} loading - Whether the app is currently loading.
 */
function setLoading(loading) {
    isLoading = loading;
    const spinner = document.querySelector('.spinner');
    const backdrop = document.querySelector('.spinner-backdrop');
    spinner.style.display = loading ? 'block' : 'none';
    backdrop.style.display = loading ? 'block' : 'none';
}

/**
 * Handle loading and preparing a new image.
 * This includes drawing it to the canvas, clearing previous points/masks,
 * and encoding it for segmentation.
 * @param {HTMLImageElement} img - The image to handle.
 */
async function handleImageLoad(img) {
    setLoading(true);
    try {
        const sourceCanvas = document.getElementById('sourceCanvas');
        const maskCanvas = document.getElementById('maskCanvas');
        const sourceCtx = sourceCanvas.getContext('2d');

        // Store the original image for resetting later
        originalImage = img;

        // Adjust canvas size to match the image
        sourceCanvas.width = maskCanvas.width = img.width;
        sourceCanvas.height = maskCanvas.height = img.height;
        imageWidth = img.width;
        imageHeight = img.height;

        // Draw the source image
        sourceCtx.drawImage(img, 0, 0);

        // Clear any previous points or masks
        points = [];
        const maskCtx = maskCanvas.getContext('2d');
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

        // Encode the image using the encoder
        embedding = await encoder.encode(img);

        // Enable the negative and reset buttons
        document.getElementById('negativeBtn').disabled = false;
        document.getElementById('resetBtn').disabled = false;
    } finally {
        setLoading(false);
    }
}

/**
 * Helper function to handle loading a test image from a given URL.
 * @param {string} url - The URL of the test image.
 * @param {string} errorMessage - The error message to show if loading fails.
 */
async function handleTestImageLoad(url, errorMessage) {
    setLoading(true);
    try {
        const img = await loadImageFromURL(url);
        await handleImageLoad(img);
    } catch (error) {
        alert(errorMessage);
        console.error(error);
    } finally {
        setLoading(false);
    }
}

/**
 * Initialize the application: set up the encoder, predictor, event listeners, etc.
 */
export async function init() {
    // Initialize the encoder and predictor models
    encoder = new SAM2Encoder();
    await encoder.initialize();

    predictor = new SAM2Predictor();
    await predictor.initialize();

    // Get references to UI elements
    const imageInput = document.getElementById('imageInput');
    const sourceCanvas = document.getElementById('sourceCanvas');
    const maskCanvas = document.getElementById('maskCanvas');
    const negativeBtn = document.getElementById('negativeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const testImage1Btn = document.getElementById('testImage1');
    const testImage2Btn = document.getElementById('testImage2');
    const sourceCtx = sourceCanvas.getContext('2d');

    // Handle user image uploads
    imageInput.addEventListener('change', async (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const img = new Image();
            img.onload = async () => {
                await handleImageLoad(img);
            };
            img.src = URL.createObjectURL(file);
        }
    });

    // Load test images
    testImage1Btn.addEventListener('click', async () => {
        await handleTestImageLoad(TEST_IMAGE_URLS.image1, 'Failed to load test image 1');
    });
    testImage2Btn.addEventListener('click', async () => {
        await handleTestImageLoad(TEST_IMAGE_URLS.image2, 'Failed to load test image 2');
    });

    // Toggle negative point mode
    negativeBtn.addEventListener('click', () => {
        isNegative = !isNegative;
        negativeBtn.textContent = `Negative Points: ${isNegative ? 'ON' : 'OFF'}`;
        negativeBtn.classList.toggle('negativeMode', isNegative);
    });

    // Reset the canvas and clear all points/masks
    resetBtn.addEventListener('click', () => {
        if (!originalImage) return;

        points = [];
        sourceCtx.drawImage(originalImage, 0, 0);
        const maskCtx = maskCanvas.getContext('2d');
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    });

    // Add points by clicking on the source canvas
    sourceCanvas.addEventListener('click', async (e) => {
        if (!embedding || isLoading) {
            alert('Please wait for the image to finish encoding.');
            return;
        }

        setLoading(true);
        try {
            const rect = sourceCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // type: 1 = Positive point, 0 = Negative point
            const point = { x, y, type: isNegative ? 0 : 1 };
            points.push(point);

            // Draw the point on the source canvas
            drawPoint(sourceCtx, point);

            // Run the segmentation prediction
            const results = await predictor.predict(embedding, points);
            const masksTensor = results['masks'];

            // Draw the generated mask
            drawMask(maskCanvas, masksTensor, imageWidth, imageHeight);
        } finally {
            setLoading(false);
        }
    });
}
