#gmake
# input 
# output : txt file
pwd=`pwd` #backtick
mkdir -p -v job_list # make dir ./job_list
confdir=GEMSTOOL_UVNONE_Configfiles
use_qsub=0
file_template=./UVNPCA_TOPLEVEL.template_lut
file=./UVNPCA_TOPLEVEL.lut
lat=15 # 85 for H, 45 for M, 15 for L
nstokes='1' # n-stokes
nss='02' # n-stream
nls='24'

# output directory
outdir='/RSL2/soodal/1_DATA/GEMSTOOL/lutdata/sca'$nss'st/'
if [ $nstokes -eq 3 ]
then
outdir='/RSL2/soodal/1_DATA/GEMSTOOL/lutdata/vec'$nss'st-h/'
fi


# lut default node setting by juseon

#tozs='200 250 300 350 400 450 500 550'          # for mid latitude
#tozs='100 150 200 250 300 350 400 450 500 550'  # for high latitude
#tozs='200 250 300 350'                          # tropics

#sfcs='1050.00 1013.25  0900.00 0800.00 0700.00 0600.00 0500.00 0400.00 0300.00 0200.00 0150.00 0100.00'
#szas='00.0 16.0 31.0 44.0 55.0 64.0 71.0 76.5 80.5 83.5 86.0 88.0'
#vzas='00.0 15.0 30.0 43.0 53.0 61.0 67.0 72.0' # 76.0 79.0 81.0 83.0 85.0 87.0 88.0 89.0'
#azas='000 020 045 090 110 135 160 180'
#albs='0.000 0.050 0.100 0.200  0.300 0.400 0.500 0.600 0.800 1.000'
#albs='0.00 0.15 0.8'

# custom options

#szas='08.0 23.5 37.5 49.5 67.5 73.75 78.5 82.0 84.75 87.0 '
szas='0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31.0 32. 33. 34. 35. 36. 37. 38. 39. 40. 41. 42. 43. 44.0 45. 46. 47. 48. 49. 50. 51. 52. 53. 54. 55.0 56. 57. 58. 59. 60. 61. 62. 63. 64.0 65. 66. 67. 68. 69. 70. 71.0 72. 73. 74. 75. 76. 76.5 78. 79. 80. 80.5 81. 82. 83. 83.5 84. 85. 86.0 87. 88.0 89. '
tozs='300'                          # tropics
sfcs='1050.0'
vzas='15.0'
azas='90'
albs='0.100'
cth=1

# qsub related
njob=0
thejob=0
jobsub=rtm3run_

# start loop
for nstoke in $nstokes
do
for alb in $albs
do
for sfc in $sfcs
do
for ns in $nss
do
for nl in $nls
do
for toz in $tozs
do
for aza in $azas
do  
for vza in $vzas
do
for sza in $szas
do
start=1
echo "sza:" $sza "vza:" $vza "aza:"$aza "toz:"$toz "nl:" $nl "pre":$sfc "alb:"$alb "NS:" $ns
echo $file

# write input file based on template
   sed \
    -e s,"@NSTREAM",$ns,g \
    -e s,"@NSTOKE",$nstoke,g \
    -e s,"@ALB",$alb,g \
    -e s,"@SPRES",$sfc,g \
    -e s,"@NL",$nl,g \
    -e s,"@TOZ",$toz,g \
    -e s,"@SZA",$sza,g \
    -e s,"@VZA",$vza,g \
    -e s,"@AZA",$aza,g \
    -e s,"@LAT",$lat,g \
    -e s,"@OUTDIR",$outdir,g \
     $file_template > $file

if [ $use_qsub -eq 1 ]
 then
 thejob=`expr $thejob + 1`
 jobdir=$jobsub$thejob
 mkdir -p -v $jobdir
 cd $jobdir
 rm -rf ./*.exe
 ln -sf ../*.exe ./
 ln -sf ../$confdir ./
 cp ../$file ./$file
 cd ../ ##???

 #echo $thejob "vza:"$vza "aza:"$aza "toz:"$toz "pre":$sfc "alb:"$alb >> job.list
 echo $thejob
 echo $thejob  >> job.list
 njob=`expr $njob + 1`
 jobs[$njob]=$thejob
 
 if [ $njob -ge $cth ]
  then 
  jobfile=`pwd`/job_list/$jobsub$thejob
  mv job.list $jobfile
  echo "qsub $njob" $jobs
  qsub -t $start-$njob -N "RTM" qsub.sh  $jobfile $jobsub $file ${jobs[@]}
  #sh qsub.sh  $pwd $jobfile $jobsub ${jobs[@]}
  jobs[0]='' ; njob=0
  rm -rf job.list
 fi
fi

if [ $use_qsub -eq 0 ]
  then
  echo aaaaaaaaaaa $jobsub $thejob $jobdir
  ./Gemstool_UVNPCA_Driver_Plus.exe  $file
fi

done
done
done
done
done
done
done
done
done
echo $njob

if [ $use_qsub -eq 1 ] 
then
 if [ $njob -gt 0 ]
  then 
  jobfile=`pwd`/job_list/$jobsub$thejob
  mv job.list $jobfile
  echo "qsub $njob" $jobs
  qsub -t 1-$njob -N "RTM" qsub.sh  $pwd $jobfile $jobsub $file ${jobs[@]}
  jobs[0]='' ; njob=0
  rm -rf job.list
 fi
fi
exit
