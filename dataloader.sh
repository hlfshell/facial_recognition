##########################
#       Data Loader      #
##########################

# The goal of this script is to download the Udacity
# Facial Recognition dataset from Github, pull only
# the dataset, and then clear out the unneeded code.

wget "https://github.com/udacity/P1_Facial_Keypoints/archive/master.zip"

unzip master.zip

rm master.zip

mv P1_Facial_Keypoints-master/data ./

rm -rf P1_Facial_Keypoints-master