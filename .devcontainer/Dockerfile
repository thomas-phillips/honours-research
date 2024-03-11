FROM ubuntu:22.04

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# Set Timezone for tzdata installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ 'UTC'

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
    && apt-get -y install sudo coreutils \
    openssh-client git git-core bash-completion \
    xxd vim-common wget curl apt-transport-https ca-certificates \
    gnupg2 software-properties-common usbutils iproute2 \
    zsh fonts-font-awesome tldr

RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.8 python3-pip nano zsh vim neovim

# Set Python 3 as default for python command.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN addgroup --gid $GROUP_ID user \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user \
    && usermod -a -G sudo user && echo "user ALL=(ALL) NOPASSWD:ALL" >>  /etc/sudoers

USER user

# RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# RUN sudo chown -R user:user /home/user
