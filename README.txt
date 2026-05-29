IA.pdf - Partea de lucrare scrisa
detalii teste.txt - notite brute facute in timpul antrenarii/testarii + outputs din consola

Fiecare folder de training contine:
	-folderul data - contine dataseturile folosite
	-folderul screenshots - contine graficele rezultate si alte imagini utile
	-folderul model - contine modelele antrenate (nu folosite in mod curent)
	-scripturi de antrenare ale modelului
	-scripturi helper (pentru tunare/testare/etc.../)

cd "Classification Training"				-- segment task clasificare

python classification_task.py  				-- rulare atrenare modele principale
python classification_task_benchmarking.py	  	-- rulare script comparare performanta
python classification_task_model_comparasion.py  	-- rulare script comparare algoritmi clasificare principali 
python classification_task_parameters_tuning.py 	-- rulare script comparare a hiperparametrilor pentru fiecare algoritm


cd "Regression Training" 				-- segment task regresie

python regression_task.py 				-- rulare antrenare modele principale
python regression_task_parameters_tuning.py		-- rulare script comparare a hiperparametrilor pentru fiecare algoritm



cd "Clustering Training"				-- segment task clusterizare

python clustering_task_stars.py				-- rulare antrenare model pentru stele
python clustering_task_countries.py			-- rulare antrenare model pentru tari


Repo: https://github.com/ClaudiuCristianRotaru/Proiect-IA

