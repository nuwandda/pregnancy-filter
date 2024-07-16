# Pregnancy Filter
<a name="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## Introduction
The stable diffusion pregnancy filter represents a pioneering advancement in image processing, offering a unique and innovative approach to visualizing the transformative journey of pregnancy through digital imagery. Leveraging the principles of stable image diffusion and artistic rendering techniques, this novel filter enables the creation of captivating and emotive representations that celebrate the beauty of pregnancy while preserving the integrity of the original image.

Pregnancy, a profound and life-changing experience, is marked by a myriad of physical and emotional changes that shape the journey to parenthood. Capturing and commemorating these moments through photography has become increasingly popular, allowing expectant parents to preserve cherished memories and share their joy with loved ones. However, traditional image filters often fall short in effectively conveying the nuanced emotions and transformations associated with pregnancy.

The stable diffusion pregnancy filter seeks to address this limitation by providing a sophisticated tool for enhancing pregnancy-related imagery in a visually striking and emotionally resonant manner. By combining stable diffusion algorithms with specialized pregnancy-themed artistic effects, this filter offers a unique means of accentuating the beauty of pregnancy while maintaining the authenticity of the original photograph.

Through subtle adjustments to color tones, texture gradients, and structural elements, the stable diffusion pregnancy filter imbues images with a sense of warmth, tenderness, and maternal radiance. Whether applied to maternity portraits, ultrasound scans, or pregnancy announcement photos, this filter enhances the visual impact of pregnancy imagery, evoking a sense of wonder and anticipation that transcends conventional image processing techniques.

Beyond its aesthetic appeal, the stable diffusion pregnancy filter holds promise for various applications across the realms of healthcare, education, and personal expression. From medical imaging and educational materials to social media content and digital artistry, this innovative filter offers a versatile tool for enhancing the visual representation of pregnancy and fostering meaningful connections between expectant parents and their communities.

As research and development in stable diffusion pregnancy filter technology continue to advance, this transformative tool is poised to redefine the way pregnancy-related imagery is perceived, experienced, and shared in the digital age. By harnessing the power of image processing and artistic expression, the stable diffusion pregnancy filter celebrates the beauty and wonder of pregnancy, enriching the visual landscape of parenthood for generations to come.

![sample_image](samples/sample.png "Sample Pregnancy Filter")

<!-- ARCHITECTURE -->
## Architecture
Stable Diffusion, a latent text-to-image diffusion model released in 2022, employs latent diffusion models (LDMs). LDMs iteratively reduce noise in a latent representation space and convert it into complete images. The text-to-image generation process involves an Image Encoder, Text Encoder, Diffusion Model, and Image Decoder. The Image Encoder and Text Encoder transform images and text into latent representations, the Diffusion Model generates new images guided by text, and the Image Decoder reconstructs images from the latent space. Stable Diffusion excels in generating detailed images from text and supports tasks like inpainting and image-to-image translations. Its weights, model card, and code are publicly available.

The model used in this project is called "Realistic Vision". Realistic Vision is an all-rounded model for generating photograph-style images. In addition to realistic people, it is also good for products and scenes. Please visit this <a href="https://civitai.com/models/4201/realistic-vision-v60-b1">link</a> to see details.
<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- Used Technologies -->
## Used technologies
### FastAPI
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints. It is designed to be easy to use, while also being fast and efficient. Some key features and advantages of FastAPI include:

* Fast and High Performance: FastAPI is built on top of Starlette and Pydantic, utilizing asynchronous programming to achieve high performance. It is one of the fastest web frameworks available for Python.

* Automatic API Documentation: FastAPI automatically generates interactive API documentation (using Swagger UI and ReDoc) based on the Python type hints, making it easy for developers to understand, test, and consume the API.

* Type Hints and Data Validation: FastAPI uses Python type hints for request and response data, enabling automatic data validation. This helps catch errors early in the development process and improves the overall reliability of the API.

* Dependency Injection System: FastAPI provides a built-in dependency injection system, making it easy to manage and inject dependencies into route functions.

* Security: It comes with built-in security features, such as OAuth and JWT token support, which simplifies the implementation of secure authentication and authorization in APIs.

* WebSocket Support: FastAPI supports WebSocket communication, allowing real-time bidirectional communication between clients and the server.

* Synchronous and Asynchronous Code: FastAPI supports both synchronous and asynchronous code, making it flexible for different use cases and allowing developers to leverage the benefits of asynchronous programming when needed.

* Easy Integration with Other Libraries: FastAPI seamlessly integrates with other popular Python libraries and frameworks, such as SQLAlchemy, Tortoise-ORM, and others.

* Automatic Generation of API Client Code: Using the generated OpenAPI documentation, FastAPI can automatically generate API client code in multiple programming languages, reducing the effort required to consume the API.

* Active Development and Community Support: FastAPI is actively developed and has a growing community. The framework is well-documented, and its community actively contributes to its improvement.

Overall, FastAPI is a modern and powerful web framework that prioritizes developer productivity, type safety, and high performance, making it an excellent choice for building APIs with Python.

### Uvicorn
Uvicorn is an ASGI (Asynchronous Server Gateway Interface) server that is specifically designed to run ASGI applications, such as those built with the FastAPI web framework. ASGI is a specification for asynchronous web servers and applications in Python, providing a standard interface between web servers and Python web applications or frameworks.

Here are some advantages of using Uvicorn:

* ASGI Support: Uvicorn supports the ASGI specification, which is designed to handle asynchronous programming and enables the development of highly concurrent web applications.

* Fast and Efficient: Uvicorn is known for its high performance and efficiency, making it well-suited for handling concurrent connections and delivering fast responses.

* Compatibility with FastAPI: Uvicorn is the recommended server for running FastAPI applications. When paired with FastAPI, it allows developers to take full advantage of asynchronous programming and achieve optimal performance.

* Ease of Use: Uvicorn is easy to install and use. It can be started with a single command, making it accessible for developers at all levels.

* WebSocket Support: Uvicorn supports WebSocket communication, allowing real-time bidirectional communication between clients and the server. This is particularly useful for applications that require real-time updates.

* Graceful Shutdown: Uvicorn supports graceful shutdowns, allowing existing requests to finish processing before the server stops. This helps maintain the stability and reliability of the application.

* Configuration Options: Uvicorn provides various configuration options, allowing developers to customize the server settings based on the requirements of their applications.

* TLS/SSL Support: Uvicorn supports TLS/SSL encryption, providing a secure way to transmit data over the network.

* Active Development and Community Support: Uvicorn is actively maintained and has a supportive community. Regular updates and contributions from the community ensure that the server stays up-to-date and improves over time.

* Integration with Other ASGI Frameworks: While commonly used with FastAPI, Uvicorn is not limited to a specific framework. It can be used with other ASGI frameworks and applications, providing flexibility and compatibility.

In summary, Uvicorn is a versatile and performant ASGI server that excels in handling asynchronous web applications. Its compatibility with FastAPI and support for WebSocket communication make it a popular choice for developers building modern, real-time web applications with Python. 

For this project, Uvicorn is using 3 workers. This means there will 3 subprocesses and the users can send requests in parallel. With this feature, the server can accept more than one request at the same time. You can increase the worker number regarding to your VRAM.

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started - Python
Instructions on setting up your project locally.
To get a local copy up and running follow these simple steps.

### Get submodules
To get the submodules, in a terminal, type:
  ```sh
  git submodule update --init --recursive
  ```

### Install dependencies
To install the required packages, in a terminal, type:
  ```sh
  pip install -r requirements.txt
  ```

### Download model
To download the model, in a terminal, type:
  ```sh
  wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/weights/realisticVisionV60B1_v20Novae.safetensors\?download\=true --directory-prefix weights --content-disposition
  ```

### Run the project
To run the project, in a terminal, type:
  ```sh
  uvicorn app:app --proxy-headers --host 0.0.0.0 --port 8000 --workers 3
  ```
Then, visit <a href="http://localhost:8000/docs">http://localhost:8000/docs</a> to see the endpoints.

## Getting Started - Docker
Instructions on setting up your project locally using Docker.
To get a local copy up and running follow these simple steps.

### Build Docker
To build the Docker image, in a terminal, type:
  ```sh
  docker build -t pregnancy_filter -f Dockerfile .
  ```

### Run the container
To run the container, in a terminal, type:
  ```sh
  docker run -it -d --gpus all --name pregnancy_filter -p 80:80 pregnancy_filter
  ```
Then, visit <a href="http://localhost/docs">http://localhost/docs</a> to see the endpoints.

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

