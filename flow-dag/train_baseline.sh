graph_type=('erdos-renyi' 'barabasi-albert')
graph_linear_type=('linear' 'nonlinear_1' 'nonlinear_2')
graph_sem_type=('linear-gauss' 'linear-exp' 'linear-exp')
graph_degree=(3 4)
data_variable_size=(10 20 50 100)

for i in {1..10}
do
    for gt in "${graph_type[@]}"
    do
        for glt in "${graph_linear_type[@]}"
        do
            for gst in "${graph_sem_type[@]}"
            do
                for j in 3 4
                do
                    python3.6 baseline.py --graph_type "$gt" --graph_linear_type "$glt" --graph_sem_type "$gst" --graph_degree $j --data_variable_size 10 
                    python3.6 baseline.py --graph_type "$gt" --graph_linear_type "$glt" --graph_sem_type "$gst" --graph_degree $j --data_variable_size 20 
                    python3.6 baseline.py --graph_type "$gt" --graph_linear_type "$glt" --graph_sem_type "$gst" --graph_degree $j --data_variable_size 50 
                    python3.6 baseline.py --graph_type "$gt" --graph_linear_type "$glt" --graph_sem_type "$gst" --graph_degree $j --data_variable_size 100 
                done
            done
        done
    done
done