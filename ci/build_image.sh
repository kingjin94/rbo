# Run this script from the mcs root directory to build the docker image for the mcs project.
docker build -t gitlab.lrz.de:5005/tum-cps/robot-base-pose-optimization/ompl:main -f ci/Dockerfile .

echo "To update the image on gitlab/docker hub, run:"
echo "docker push gitlab.lrz.de:5005/tum-cps/robot-base-pose-optimization/ompl:main"
