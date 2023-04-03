ARG DOKCER_CUDA_VERSION
FROM nvidia/cuda:${DOKCER_CUDA_VERSION}


# environmental variable used in the process of build
ARG DOCKER_USER
ARG DOCKER_UID
ARG DOCKER_GID
ARG DOCKER_PASS
ARG DOCKER_GROUPNAME
ARG HOME=/home/$DOCKER_USER


# enviroment variable setted in container
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ Asia/Tokyo

#change repository server to the server in Japan
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
#update & upgrade
RUN apt-get -y update && apt-get -y upgrade

#install something useful
RUN apt-get install -y \
    sudo tzdata git python3 python3-pip
    
# set user
RUN groupadd -g $DOCKER_GID $DOCKER_GROUPNAME && \
    useradd -d $HOME -m -s /bin/bash -u $DOCKER_UID -g $DOCKER_GID $DOCKER_USER && \
    echo "$DOCKER_USER:$DOCKER_PASS" | chpasswd && \
    echo "$DOCKER_USER ALL=(ALL) ALL" >> /etc/sudoers.d/$DOCKER_USER
USER $DOCKER_USER

WORKDIR $HOME

#after making docker container, do passwd to change your password