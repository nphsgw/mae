FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    git wget libgl1-mesa-glx libglib2.0-0 \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add user
ARG USERNAME=developer
ARG GROUPNAME=localuser
ARG UID=1000
ARG GID=1000
ARG PASSWORD=pass
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME

RUN pip install jupyter black flake8 mypy numpy pandas matplotlib scikit-learn opencv-python torchinfo torchsummary torchtyping
RUN pip install tensorboard timm submitit six