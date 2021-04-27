DOCKERIMAGE="bmf_control"
DOCKERNAME="BMFcontroller"
CREDENTIAL_CACHE_PREFIX="FILE:"

klist --test
if [ $? -ne 0 ] ; then
   kinit
   if [ $? -ne 0 ] ; then
      echo "Could not authenticate..."
      exit 1
   fi
fi
CREDENTIAL_CACHE=${KRB5CCNAME#"$CREDENTIAL_CACHE_PREFIX"}
docker run --name $DOCKERNAME --rm \
--net=host \
-v /var/run/docker.sock:/var/run/docker.sock \
-v /media/scratch/nesser/Projects:/media/scratch/nesser/Projects \
-v /beegfsEDD/NESSER:/beegfsEDD/NESSER \
-v /etc/krb5.conf:/etc/krb5.conf \
-v $CREDENTIAL_CACHE:$CREDENTIAL_CACHE \
-e USER=root \
-it $DOCKERIMAGE /bin/bash -ic "export KRB5CCNAME=$CREDENTIAL_CACHE;cd bmf_bypass_controlling && git pull;bash" \
