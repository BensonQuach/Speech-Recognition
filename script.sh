for i in {1..15} 
do
head -n 1 "unkn_$i.lik"
done

submit -f -N A3 csc401h *