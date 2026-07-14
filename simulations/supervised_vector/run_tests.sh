#!/bin/bash
for input_type in vector_only visual_only both random
do
   for i in {1..10}
   do
      python test_agent.py $i $input_type
   done
done


