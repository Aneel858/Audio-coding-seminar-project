# Step 1: Update locale packages and generate en_US.UTF-8
!sudo apt-get update -y
!sudo apt-get install locales -y
!sudo locale-gen en_US.UTF-8

# Step 2: Export locale environment variables
%env LANG=en_US.UTF-8
%env LANGUAGE=en_US:en
%env LC_ALL=en_US.UTF-8

# Just to confirm
!echo "Locale is now: $LANG"
