#!/bin/bash
while read -r ARGSTR; do
  # There are some blank lines in coords.txt, so check that the line isn't empty!
  # Obviously a good next step would be to validate the fields to be ultra defensive (just good practise!)
  if [ ${#ARGSTR} -eq 0 ]; then
    echo "This line has no obvious data..."
    continue
  fi

  FILENAME=`echo $ARGSTR | awk -F, '{ print $2 }'`
  echo $FILENAME
  COORDS=`echo $ARGSTR | awk -F, '{ print $3 " " $4 " " $5 " " $6 }'`
  DATE=`echo $ARGSTR | awk -F, '{print $1}'`
  echo $DATE
  sed -r "s/DATE/$DATE/g" bands367_3031.xml >template.xml
  gdal_translate -of GTiff -projwin $COORDS template.xml $FILENAME
done < coords.txt