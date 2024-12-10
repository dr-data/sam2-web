const DECODER_MODEL_URL = 'https://storage.googleapis.com/lb-artifacts-testing-public/sam2/sam2_hiera_tiny.decoder.onnx';
const SAM_ONNX_MASK_SIZE = 256;

class SAM2Predictor {
    constructor() {
        this.session = null;
    }

    async initialize() {
        try {
            console.log('Loading decoder model...');
            this.session = await ort.InferenceSession.create(DECODER_MODEL_URL);
            console.log('Decoder model loaded successfully');
        } catch (e) {
            console.error('Failed to load decoder model:', e);
            throw e;
        }
    }

    getPointScaleFactor(imageHeight, imageWidth) {
        const longSide = Math.max(imageHeight, imageWidth);
        const imageScale = 1024 / longSide;
        const factors = {
            x: imageScale,
            y: imageScale,
        };
        if (imageHeight < imageWidth) {
            factors.y *= imageWidth / imageHeight;
        } else {
            factors.x *= imageHeight / imageWidth;
        }
        return factors;
    }

    prepareInputs(embedding, points) {
        const inputs = {};

        inputs['image_embed'] = embedding;

        // Number of labels (batch size) is 1
        const numLabels = 1;
        const numPoints = points.length;

        // Prepare 'point_coords' and 'point_labels'
        const pointCoordsData = [];
        const pointLabelsData = [];

        for (let point of points) {
            pointCoordsData.push([point.x, point.y]);
            pointLabelsData.push(point.type);
        }

        // Ensure the tensors have the correct shapes
        inputs['point_coords'] = new ort.Tensor(
            'float32',
            Float32Array.from(pointCoordsData.flat()),
            [numLabels, numPoints, 2]
        );

        inputs['point_labels'] = new ort.Tensor(
            'float32',
            Float32Array.from(pointLabelsData),
            [numLabels, numPoints]
        );

        // Prepare 'mask_input' with zeros
        inputs['mask_input'] = new ort.Tensor(
            'float32',
            new Float32Array(numLabels * 1 * SAM_ONNX_MASK_SIZE * SAM_ONNX_MASK_SIZE),
            [numLabels, 1, SAM_ONNX_MASK_SIZE, SAM_ONNX_MASK_SIZE]
        );

        // Prepare 'has_mask_input' tensor
        inputs['has_mask_input'] = new ort.Tensor(
            'float32',
            new Float32Array([0.0]), // 0.0 indicates no prior mask
            [numLabels]
        );

        // Prepare 'high_res_feats_0' and 'high_res_feats_1' with zeros
        inputs['high_res_feats_0'] = new ort.Tensor(
            'float32',
            new Float32Array(1 * 32 * 256 * 256),
            [1, 32, 256, 256]
        );

        inputs['high_res_feats_1'] = new ort.Tensor(
            'float32',
            new Float32Array(1 * 64 * 128 * 128),
            [1, 64, 128, 128]
        );

        return inputs;
    }

    async predict(embedding, inputPoints) {
        try {
            // Prepare inputs for the decoder model
            const inputs = this.prepareInputs(embedding, inputPoints);

            // Run inference
            const results = await this.session.run(inputs);

            return results;
        } catch (e) {
            console.error('Prediction error:', e);
            throw e;
        }
    }
}

export default SAM2Predictor;