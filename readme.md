# SAM2 Web Demo

This repository contains a demo project for the article:

**Bringing AI to the browser: Running SAM2 for interactive image segmentation**

The project demonstrates how to use the Segment Anything Model 2 (SAM2) for interactive image segmentation directly in the web browser using ONNX Runtime Web (`ort`). It allows users to upload images, interactively add positive or negative points, and see the segmentation mask update in real-time—all running entirely client-side.

## Features

- **Client-Side Execution**: All computations are performed in the browser, ensuring user privacy and a responsive user experience.
- **Interactive Segmentation**: Users can refine segmentation results by adding positive (foreground) or negative (background) points.
- **ONNX Runtime Web Integration**: Leverages `ort` to run the SAM2 encoder and decoder models efficiently in the browser.

## Article

For a detailed explanation of how this project works, please refer to the accompanying article:

[Bringing AI to the browser: Running SAM2 for interactive image segmentation](link_to_article)

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [ONNX Runtime Web](https://onnxruntime.ai/docs/api/javascript/index.html)
- [Segment Anything Model 2 (SAM2)](https://segment-anything.com/)
