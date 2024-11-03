

# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

#FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set up environment variable, Important to use gpu in the container 
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# Set environment variables
#ENV WANDB_API_KEY="<TYPE YOUR KEY VALUE>"

RUN apt-get update && apt-get install -y graphviz

COPY ./requirements.txt /install/requirements.txt
RUN pip3 install -r /install/requirements.txt

WORKDIR /code