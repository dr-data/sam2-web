const ENCODER_MODEL_URL = 'https://storage.googleapis.com/lb-artifacts-testing-public/sam2/sam2_hiera_tiny.encoder.ort';

class SAM2Encoder {
    constructor() {
        this.session = null;
        this.lastEmbeddings = null;
    }

    async initialize() {
        try {
            console.log('Starting to load encoder model...');
            // Create session
            this.session = await ort.InferenceSession.create(ENCODER_MODEL_URL);
            console.log('Encoder session created successfully');
        } catch (e) {
            console.error('Failed to load encoder model:', e);
            throw e;
        }
    }

    async encode(image) {
        try {
            // Prepare input tensor
            const tensor = this.imageDataToTensor(image);

            // Run inference
            const feeds = { image: tensor };
            const results = await this.session.run(feeds);

            // Store the embeddings
            this.lastEmbeddings = results.image_embed;

            return this.lastEmbeddings;
        } catch (e) {
            console.error('Encoding error:', e);
            throw e;
        }
    }

    imageDataToTensor(image) {
        // Resize and normalize the image
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 1024;
        canvas.height = 1024;

        // Draw and resize the image
        ctx.drawImage(image, 0, 0, 1024, 1024);
        const imageData = ctx.getImageData(0, 0, 1024, 1024).data;

        // Convert ImageData to float32 tensor normalized to [-1, 1]
        const inputArray = new Float32Array(3 * 1024 * 1024);

        for (let i = 0; i < 1024 * 1024; i++) {
            inputArray[i] = (imageData[i * 4] / 255.0) * 2 - 1;                     // R
            inputArray[i + 1024 * 1024] = (imageData[i * 4 + 1] / 255.0) * 2 - 1;  // G
            inputArray[i + 2 * 1024 * 1024] = (imageData[i * 4 + 2] / 255.0) * 2 - 1; // B
        }

        return new ort.Tensor('float32', inputArray, [1, 3, 1024, 1024]);
    }
}

export default SAM2Encoder;
