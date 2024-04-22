How to use the programs:
Inside the CICIDS or SIMARGL or NSLKDD folder you will find programs for each model used in this paper. Each one of these programs outputs: (Note: For the CNN programs refer to CNN folder)

The accuracy for the AI model.
The values for y_axis for the sparsity (that will need to be copied and pasted into the general_sparsity_graph.py).
Top features in importance order (that will be needed to rerun these same programs to obtain new Accuracy values for Descriptive Accuracy. Take note of the values as you use less features and input these values in general_desc_acc_graph.py ).
For Stability, run the programs 3x or more and input the obtained top k features in general_stability_comparison.py).
Descriptive Accuracy:

To generate Descriptive Accuracy Graphs, see the code general_desc_acc_graph.py
Sparsity:

To generate Sparsity Graphs, see the code general_sparsity_graph.py
Stability:

To generate the Stability metrics, see the code general_stability_comparison.py
Robustness:

Inside the Robustness folder, firts run the code: threshold_CIC.py to generate a csv file. Then run analyze_threshold_CIC_LIME.py to generate the Robustness Sensitivity graph. and run the program RF_SHAP_CIC_bar.ipynb to generate the robustness bar graphs.
Completeness:

Inside the CICIDS or SIMARGL or NSLKDD folder run the code RF_LIME_COM_SML_CHART.ipynb or RF_SHAP_COM_SML_CHART.ipynb as an example.
Efficiency:

Inside the CICIDS or SIMARGL or NSLKDD folder you will find programs for each model used in this paper. They output the time spent to generate the SHAP or LIME evaluation for k samples. We can just set up in the program the k value and take note of the time spent.
