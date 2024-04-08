# IIsy: Practical & Hybrid In-network ML Classification

IIsy introcuces hybrid in-network ML deployment which employs a small in-network ML model on the network device and a large ML model over the end-point. 💡**IIsy's model mapping for in-network ML is now integrated in Plnater [link].**

<img src="src/IIsy.png" width = "500"  align= left/>

## How to test the hybrid model

- Change the dataset. Making sure the new data loader has the ```load_data``` function.
		
     ```
     vim src/<dataset name>_dataset.py
     ```
     Open the main file by using ```vim main.py``` and change the load data name code such as:
     
	  ```
	  from src.Iris_dataset import load_data
	  ...
	  features_subset = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
	  ...
	  X_train, y_train, X_test, y_test, used_features = load_data(4, './Data')
	  ```
    
- Change the parameters (e.g., model size, depth, and used features) for the small in-network ML model on the network device and a large ML model over the end-point.
		
     ```
     vim main.py
     ```
- Run the simulation.
	
    ```
    python3 main.py
    ``` 
    The algorithm will output the switch fraction and accuracy of both model and syatem under different confidence thresholds.
    
    ```
    ...
    Switch confidence th=0.64: AUC 0.981143 , Macro-F1  0.963309 , Accuracy 0.984522 , Precision 0.983752 , Recall 0.998776 , Precision normal switch 0.985413 , Precision normal server 0.999157  , Precision anomaly switch 0.000000 , Precision anomaly server 0.912983 , Switch Fraction 0.848257
    Switch confidence th=0.66: AUC 0.981143 , Macro-F1  0.963309 , Accuracy 0.984522 , Precision 0.983752 , Recall 0.998776 , Precision normal switch 0.985413 , Precision normal server 0.999157  , Precision anomaly switch 0.000000 , Precision anomaly server 0.912983 , Switch Fraction 0.848257
    Switch confidence th=0.68: AUC 0.981144 , Macro-F1  0.963354 , Accuracy 0.984540 , Precision 0.983774 , Recall 0.998774 , Precision normal switch 0.985442 , Precision normal server 0.999164  , Precision anomaly switch 0.000000 , Precision anomaly server 0.912990 , Switch Fraction 0.848227
    Switch confidence th=0.70: AUC 0.981144 , Macro-F1  0.963354 , Accuracy 0.984540 , Precision 0.983774 , Recall 0.998774 , Precision normal switch 0.985442 , Precision normal server 0.999164  , Precision anomaly switch 0.000000 , Precision anomaly server 0.912990 , Switch Fraction 0.848227
    ...
    ``` 
    
    
## Reporting a Bug
Please submit an issue with the appropriate label on [Github](../../issues).

## License

The files are licensed under Apache License: [LICENSE](./LICENSE). The text of the license can also be found in the LICENSE file.

## Citation
If you use this code, please cite our [papers](https://dl.acm.org/doi/abs/10.1145/3472716.3472846):

```
@article{zheng2024automating,
  title={{Planter: Rapid Prototyping of In-Network Machine Learning Inference}},
  author={Zheng, Changgang and Zang, Mingyuan and Hong, Xinpeng and Perreault, Liam and Bensoussane, Riyad and Vargaftik, Shay and Ben-Itzhak, Yaniv and Zilberman, Noa},
  journal={ACM SIGCOMM Computer Communication Review},
  year={2024}
}

@article{zheng2024iisy,
  title={{IIsy: Hybrid In-Network Classification Using Programmable Switches}},
  author={Zheng, Changgang and Xiong, Zhaoqi and Bui, Thanh T and Kaupmees, Siim and Bensoussane, Riyad and Bernabeu, Antoine and Vargaftik, Shay and Ben-Itzhak, Yaniv and Zilberman, Noa},
  journal={IEEE/ACM Transactions on Networking},
  year={2024}
}

@ARTICLE{10365500,
  title={{In-Network Machine Learning Using Programmable Network Devices: A Survey}}, 
  author={Zheng, Changgang and Hong, Xinpeng and Ding, Damu and Vargaftik, Shay and Ben-Itzhak, Yaniv and Zilberman, Noa},
  journal={IEEE Communications Surveys & Tutorials}, 
  year={2023},
  doi={10.1109/COMST.2023.3344351}
}

@incollection{zheng2021planter,
  title={Planter: seeding trees within switches},
  author={Zheng, Changgang and Zilberman, Noa},
  booktitle={Proceedings of the SIGCOMM'21 Poster and Demo Sessions},
  pages={12--14},
  year={2021}
}
```

