FROM gitlab.lrz.de:5005/tum-cps/mcs/ompl:main
MAINTAINER Matthias Mayer <matthias.mayer@tum.de>
COPY . /home/base_opt
RUN cd /home/base_opt && pip install -r requirements.txt
WORKDIR /home/base_opt
