#Code
Code.py contains the entire code for the plots and costs calculations for this assignment.

There are two functions in this code namely random_kmeans(d,k) , and plus_plus(d,k)  and kmeans(u,d,k). random_kmeans(d,k) , and plus_plus(d,k) are meant to initialize the centers and call kmeans.
Both functions return labels(an array with the labels), u(the means or centers) and cost.
For obtaining the data for the assignment each function was called 5 times per k and calculated the minimum
Although, this can take really long (about 45 minutes for 8 values of k) so the loop in the functions is now set to 1 in case it needs testing.
Most of the commented lines were meant to generate the plots for the report.
A simple call to this functions is labels,centers,cost = plus_plus(data,2) for 2 centers

Libraries used were: pandas,matplotlib,scipy,scikit-learn.
#Plots
Plots contain the images for the report
