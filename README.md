T.C.
SAKARYA ÜNİVERSİTESİ
BİLGİSAYAR VE BİLİŞİM BİLİMLERİ FAKÜLTESİ







ISE 402 THESIS








SENSOR DATA ANALYSİS

 

B151200003 – Berk DOĞUŞ
B151200008 – Kaan Orhun KANAL


Department

Advisor	:

:	INFORMATION SYSTEMS ENGINEERING
Prof. Dr. İsmail Hakkı CEDİMOĞLU







2019-2020 Bahar Dönemi  

T.C.
SAKARYA ÜNİVERSİTESİ
BİLGİSAYAR VE BİLİŞİM BİLİMLERİ FAKÜLTESİ








SENSOR DATA ANALYSIS






ISE 402 - THESIS

Berk DOĞUŞ
Kaan Orhun KANAL





Fakülte Anabilim Dalı	:	BİLİŞİM SİSTEMLERİ MÜHENDİSLİĞİ (INFORMATION SYSTEMS ENGINEERING)



		

FOREWORD



At the age of ‘Internet of Things’, sensors have become more important than it was before. Collecting and analyzing sensor data in a right way is crucial for developing. As taking this situation into account, we have looked for the sensor data that we can collect and analyze. Thereafter, we have decided to collect and analyze accelerometer, gyroscope and GPS sensor data from our mobile phones by developed an Android application.

We would like to thank our esteemed advisor Prof. Dr İsmail Hakkı CEDİMOĞLU who helped us during our work. 
CONTENTS



FOREWORD.....................................................................................................	iii
CONTENTS…...................................................................................................	iv
FIGURES LİST……..........................................................................................	vi
ABSTRACT.......................................................................................................	 ix
	
CHAPTER 1.	ENTRY	10
1.1.	Sensor Data	10
1.1.1.	Accelerometer sensor data	10
1.1.2.	Gyroscope sensor data	10
1.1.3.	Global positioning system (GPS) sensor data	10
CHAPTER 2.	DATA COLLECTION	12
2.1.	MIT App Inventor	12
2.2.	Android App To Collect Data	14
CHAPTER 3.	DATA ANALYSIS TOOLS	16
3.1.	Python Data Science Libraries	16
3.1.1.	Pandas	16
3.1.2.	Sklearn	16
3.1.3.	Pandas profiling	16
3.1.4.	Matplotlib	16
3.1.5.	Folium	17
3.1.6.	Seaborn	17
3.1.7.	Tkinter	17
3.1.8.	Dask	17
3.2.	Data Analysis Methods	17
3.2.1.	Linear regression	17
3.2.2.	Decision tree	18
3.2.3.	Ridge regression	18
3.2.4.	Random forest	18
3.3.	Code Optimization	18
3.4.	Google Colab	18
3.5.	The Works Mentioned but Have not Used in Project	19
3.5.1.	Works with dask library	19
3.5.2.	Google colab work	21
CHAPTER 4.	DATA ANALYSIS WITH MACHINE LEARNING	25
4.1.	Graphical User Interface	31
4.2.	CPU Times	33
4.3.	Outputs of Sitting Dataset Analysis	36
4.4.	Outputs of Walking Dataset Analysis	39
4.5.	Output of Running Dataset Analysis	42
CHAPTER 5.	CONCLUSION	45

	

 
ŞEKİLLER LİSTESİ



Figure 2.1.	MIT Inventor Logo………...……………………………...	12
Figure 2.2.	Designer Section of MIT App Inventor…………………...	  13
Figure 2.3.	Blocks Editor Section of MIT App Inventor………………	13
Figure 2.4.	User Interface of the app we created for collecting sensor data……………………………………………………..	14
Figure 2.5.	Block logic of the app we created for collecting sensor data	15
Figure 3.1.	Importing Dask Library……………………………………	19
Figure 3.2.	Dask codes2….……………………………………………..	19
Figure 3.3.	Dask codes3.………………………………………………..	19
Figure 3.4.	Dask codes4………………………………………………...	20
Figure 3.5.	Dask codes5………………………………………………...	20
Figure 3.6.	Dask codes6………………………………………………...	20
Figure 3.7.	Dask codes7………………………………………………...	21
Figure 3.8.	Dask codes8………………………………………………...	21
Figure 3.9.	Google Colab codes...……………………………………...	22
Figure 3.10.	Google Colab codes 2……………………………………...	22
Figure 3.11.	Google Colab codes 3……………………………………...	22
Figure 3.12.	Google Colab codes 4……………………………………...	23
Figure 3.13.	Google Colab codes 5……………………………………...	23
Figure 3.14.	Google Colab codes 6……………………………………...	24
Figure 3.15.	Google Colab codes 7……………………………………...	24
Figure 4.1.	Importing libraries..........…………………………………...	25
Figure 4.2.	GUI codes…...……………………………………………...	26
Figure 4.3.	GPS data to showing on map……………….……………...	26
Figure 4.4.	Reading data……..………………………………………...	27
Figure 4.5.	Random Forest Regression codes…..……………………..	27
Figure 4.6.	Random Forest Regression codes2………………………..	27
Figure 4.7.	Ridge Regression codes……………………………………	28
Figure 4.8.	Rdige Regression codes2…………………………………..	28
Figure 4.9.	Linear Regression…………………………………………..	29
Figure 4.10. 	Decision Trees Regression…………………………………	29
Figure 4.11	Decision Trees Regression codes2…………………………	30
Figure 4.12.	Exploratory data analysis codes……………………………	30
Figure 4.13.	GUI…………………………………………………………	31
Figure 4.14.	GUI-2………………………………………………………	31
Figure 4.15.	GUI-3………………………………………………………	32
Figure 4.16.	Ridre Regression scores……………………………………	32
Figure 4.17.	Ridge Regression Graphic………………………………….	33
Figure 4.18.	CPU time calculating………………………………………	33
Figure 4.19.	CPU times for exploratory data analysis…………………...	34
Figure 4.20.	CPU times for Linear Regression…………………………..	34
Figure 4.21.	CPU times for Decision Trees……………………………...	34
Figure 4.22.	CPU times for Random Forest Regression………………...	35
Figure 4.23.	CPU times for Ridge Regression…………………………..	35
Figure 4.24.	CPU times for showin GPS data on map…………………..	35
Figure 4.25.	Sitting dataset graphic……………………………………...	36
Figure 4.26.	Sitting dataset seaborn graphic……………………………..	36
Figure 4.27.	Sitting dataset linear regression analysis…………………...	37
Figure 4.28.	Decision trees regression output of sitting dataset…………	37
Figure 4.29.	Random Forest Regression output of sitting dataset……….	38
Figure 4.30.	Ridge Regression output of sitting dataset…………………	38
Figure 4.31.	Walking data graphic………………………………………	39
Figure 4.32.	Walking data graphic-2…………………………………….	39
Figure 4.33.	Linear regression output of walking dataset……………….	40
Figure 4.34.	Decision trees regression output of walking dataset……….	40
Figure 4.35.	Random forest regression output of walking dataset………	41
Figure 4.36.	Ridge regression output of walking dataset………………..	41
Figure 4.37.	Running dataset graphic……………………………………	42
Figure 4.38.	Running dataset graphic-2………………………………….	42
Figure 4.39.	Linear regression output of running dataset………………..	43
Figure 4.40.	Decision trees regression output of running dataset………..	43
Figure 4.41.	Random forest regression output of running dataset……….	44
Figure 4.42.	Ridge regression output of running dataset………………...	44
		
		
		
		
		

 
ABSTRACT



Key words: Sensor Data, Machine Learning, Data Analysis, Regression

Sensor data is the output of a device that detects some type of input from the physical environment. The aim of our study is to analyses accelerometer sensor data, gyroscope sensor data and GPS sensor data which were collected with the usage of a smartphone to detect a user’s activities. In order to collect sensor data, an Android application was built by using MIT App Inventor. 

User’s activities were examined in three condition; sitting, walking and running. The differences of all three condition has been observed and exploratory data analysis has been made. Afterwards, regression analysis has been applied to gyroscope sensor data and accelerometer sensor data with machine learning techniques(linear regression, decision trees, random forest and ridge regression). Collected GPS data has been shown in a map.

To bring all these works together, we have created a graphical user interface. Users can reach the works easier with GUI.  

CHAPTER 1.	 ENTRY


Sensor data are getting more important with the developing technology. Data analysis methods are useful to understand sensor data. We used Python language to analyze accelerometer, gyroscope and GPS data.

1.1. 	Sensor Data 

Sensors are used to detect about any physical element. Sensor data is the output of a device that detects and responds to some type of input from the physical environment. Also, sensor data is an essential component of the Internet of Things environment.

1.1.1.	 Accelerometer sensor data

The rate of change of velocity of the body with respect to time is called acceleration and Accelerometers are devices which measures acceleration. Accelerometers can measure two-dimensional and three-dimensional forms.[1]

1.1.2.	 Gyroscope sensor data

In order to get more accurate motion sensing, Gyroscope sensors are combined with Accelerometer sensors. Besides getting angular velocity data. Gyroscope sensors can also measure the motion of the object. With Gyroscope sensor in mobile phone, we can detect gestures and motion with our mobile phones.

1.1.3.	 Global positioning system (GPS) sensor data

GPS is a satellite navigation system that detect location of users. GPS provides continuous real time, three-dimensional positioning and navigation worldwide. The working of GPS is based on the “trilateration” mathematical principle. GPS receiver takes the information from the satellite and uses the method of triangulation to determine a user’s exact position. 
CHAPTER 2.	 DATA COLLECTION

For data collection, we need to create an android application which can collect data and save it. In order to create that application, MIT App Inventor is used.

2.1. 	MIT App Inventor

The application was made available through request on July 12, 2010 and released publicly on December 15, 2010. The App Inventor team was led by Hal Abelson and Mark Friedman.[2]
Figure 2.1. MIT App Inventor logo

MIT App Inventor help you to develop Android applications by using a web browser. MIT App Inventor servers store creators’ works. 

App Inventor has two sections which called The App Inventor Designer and The App Inventor Blocks Editor.










Designer section is the part that creators select the components for their app.
 Figure 2.2 Designer Section of MIT App Inventor

The Blocks Editor section is the part that creators assemble program blocks which specify how the components should behave.

 Figure 2.3 Blocks Editor Section of MIT App Inventor



2.2. 	Android App To Collect Data

By using MIT App Inventor, we built an Android application helps us to collect and save accelerometer data, gyroscope data and GPS data.
Figure 2.4 User Interface of the app we created for collecting sensor data

The app we created is seen the photo above. When user push the ‘Veri Almaya Başla’ button, the app starts to collect and save data. If user push the ‘Veri Almayi Durdur’ button, the app stops data collection.








The block of app can be seen below. It shows the logic of the app created.
 Figure 2.5 Block logic of the app we created for collecting sensor data

 
CHAPTER 3.	DATA ANALYSIS TOOLS

3.1. 	Python Data Science Libraries

To data analyze in python language, data scientists need libraries.

3.1.1.	Pandas

Pandas is a software library written for the Python programming language for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series.

3.1.2.	Sklearn
Sklearn (Scikit-learn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.[3]
3.1.3.	Pandas profiling
Pandas profiling provides analysis like type, unique values, missing values, quantile statistics, mean, mode, median, standard deviation, sum, skewness, frequent values, histograms, correlation between variables, count, heatmap visualization.
3.1.4.	Matplotlib

Matplotlib is a Python library which is using for 2D plotting. With using matplotlib, plots, histograms, power spectra, bar charts, error charts, scatterplots etc. can be generated easily.[4]



3.1.5.	Folium

Folium library is a data visualization library for Pyhton programming language. It makes it easy to visualize data on an intereactive leaflet map. It provides binding of data to a map for choropleth visualizations. Folium supports Image, Video, GeoJSON and TopoJSON overlays.[5]

3.1.6.	Seaborn

Seaborn is a Python library that based on matplotlib. Seaborn library is used to ease the challenging task of data visualization. Seaborn provides users to the creation of statistical graphics.

3.1.7.	Tkinter

Tkinter is a GUI library for Python. It provides lots of widgets such as buttons, labels, text boxes, scrollbar etc. in a GUI application.

3.1.8.	Dask

Dask library is a useful python library to work with large datasets. It is known as a pyhton paralel computing library. Dask library has a lot similarities with pandas library so that the users working with pandas library can easily work with dask library too. Unlike pandas library, dask library provides a much faster way to handle large and big data in python.[6]

3.2. 	Data Analysis Methods

3.2.1.	Linear regression

Linear regression provides to users predict a dependent variable value based on a given independent variable. By using regression technique users can find a linear relationship between input and output.[7]
3.2.2.	Decision tree

Decision tree is a supervised learning algorithm. It repeatedly splits the dataset into two more sub-nodes according to parameters. Decision trees consist leaves and decision nodes. The data is split on decision nodes and leaves are the decisions.[8]

3.2.3.	Ridge regression

Ridge regression is regression technique that helps reduce the multicollinearity of variables in models. It also is used to quantify the overfitting of the data through measuring the magnitude of coefficients.[9]

3.2.4.	Random forest

Random forest is a supervised learning algorithm. Random forest algorithm combines multiple algorithms of the same type multiple decision trees. It can be used for both regression and classification tasks.[10]

3.3. 	Code Optimization

Nowadays, efficiency has become more crucial for software programs. To have a more efficient software, code optimization is an useful method. Optimized programs can consume less memory and executes more rapidly. With limited resources in terms of computing power or memory, code optimization would make sure that we can make do with the resources we have.[11]

3.4. 	Google Colab

Google Colab is a cloud based Jupyter notebook environment which allows users to train their models on CPUs, GPUs and TPUs for free. Training models can take a lot of hours on CPUs but GPUs and TPUs, provided by Google Colab, can easily execute in a short time.

3.5. 	The Works Mentioned but Have not Used in Project

3.5.1.	 Works with dask library

The command of reading csv files is the same with pandas library.
Figure 3.1. importing Dask library

Dask library split the large dataframes to partitions to work with them efficiently, we can see the partitions count with df command just because our sensor datas not large enough to split, dask library doesn’t need to split it.
Figure 3.2. dask codes2
Unlike pandas library, you need to use compute() command to execute exploratory data analysis tools.
Figure 3.3. dask codes3

Figure 3.4. dask codes 4

To showing the partitions function of dask library, we have found a larger dataset.
 
Figure 3.5. dask codes 5

As it seen with shape command df1 dataset is much larger than our sensor dataset df1.
 
Figure 3.6. dask codes 6

Dask library split the df1 dataset to two partitions to work with the df1 dataset faster way.
 Figure 3.7. dask codes 7

We can see each of the partitions with df1.partitions[indexcode].compute() command.
  Figure 3.8. dask codes 8

3.5.2.	Google colab work

Google Colab’s Jupyter Notebook environment provided us a much stronger CPUs than our own CPUs so it’s very helpful for saving time. Furthermore, we can say that it’s so useful to make a group work.

Google Colab notebook has some exclusive python libraries. We have used google.colab libraries to read datasets which is uploaded to Google Drive. We have moved our linear regression, random forest regression, decision trees regression and ridge regression analysis works to Google Colab environment.

As an example of our works Random Forest Regression;
 Figure 3.9. Google Colab codes
 Figure 3.10. Google Colab codes 2
Figure 3.11. Google Colab codes 3
 Figure 3.12. Google Colab codes 4
 Figure 3.13. Google Colab codes 5
 Figure 3.14. Google Colab codes 6
 Figure 3.15. Google Colab codes 7





 

CHAPTER 4.	DATA ANALYSIS WITH MACHINE LEARNING

In order to optimizating our code, we have imported required libraries.
Figure 4.1. Importing libraries

We've used tkinter library for creating graphical user interface. We have 2 dropdown list item. One of them is for selecting file. The other one for selecting function that applying to dataset. On interface we have 2 buttons. The one named "Calculate Selected" for getting all variables that we needed and start calculation. The other one is for show gps datas on map.
Figure 4.2. GUI codes
Figure 4.3. GPS data to showing on map

Figure 4.4. Reading data

We've optimized and speed up our python code, it has become more readable and agile. Code optimization have made work very useful for other function and dataset varieties.
Figure 4.5. Random Forest Regression codes
Figure 4.6. Random Forest Regression codes2




Figure 4.7. Ridge Regression Codes
Figure 4.8. Ridge Regression Codes2
 Figure 4.9. Linear Regression
 Figure 4.10. Decision Trees Regression Codes
 Figure 4.11. Decision Trees Regression codes2
 Figure 4.12. Exploratory Data Analysis codes













4.1. 	Graphical User Interface
 
The created GUI provide users to select the dataset(sitting, walking or running) and the regression method(Random Forest Regression, Decision Trees Regression, Linear Regression or Ridge Regression) or exploratory data analysis then, execute selected regression method on selected data. 
Figure 4.13. GUI
Figure 4.14. GUI-2

Figure 4.15. GUI-3

When the users completed their selections, they can execute it with using calculate selected button. On our example; user has selected ‘OtururkenVeri2’ and ‘Ridge Regression’. After using calculate button, user will see the outputs which is seen on the pictures below.
Figure 4.16. Ridge Regression scores
 
Figure 4.17. Ridge Regression Graphic


4.2. 	CPU Times

The code provides us to see CPU time.
Figure 4.18. CPU time calculation









CPU times for Exploratory Data Analysis
Figure 4.19. CPU times for exploratory data analysis

CPU times for Linear Regression
Figure 4.20. CPU times for Linear Regression

CPU times for Decision Trees Regression:
Figure 4.21. CPU times for Decision Trees


CPU times for Random Forest Regression

Figure 4.22. CPU times for Random Forest Regression

CPU times for Ridge Regression

Figure 4.23. CPU times for Ridge Regression

CPU times for GPS data on Map

Figure 4.24. CPU times for GPS data on map











4.3. 	Outputs of Sitting Dataset Analysis

Sitting Dataset Graphic
Figure 4.25. Sitting Dataset Graphic

Figure 4.26. Sitting dataset seaborn graphic









Linear Regression output of sitting dataset

Figure 4.27. Sitting Dataset Linear Regression analysis

Decision Trees Regression output of sitting dataset
Figure 4.28. Decision Trees Regression output of sitting dataset

Random Forest Regression output of sitting dataset


Figure 4.29. Random Forest Regression output of sitting dataset

Ridge Regression output of sitting dataset
Figure 4.30. Ridge Regression output of sitting dataset
4.4. 	Outputs of Walking Dataset Analysis

Walking Dataset Graphics
 
Figure 4.31. Walking Data Graphic
Figure 4.32. Walking Data Graphic2










Linear Regression output of walking dataset
Figure 4.33. Linear regression output of walking dataset

Decision Regression output of walking dataset
Figure 4.34. Decision Trees regression output of walking dataset









Random Forest Regression output of walking dataset
Figure 4.35. Random Forest Regression output of walking dataset

Ridge Regression output of walking dataset
Figure 4.36. Ridge Regression output of walking dataset

4.5. 	Output of Running Dataset Analysis

Running Dataset Graphics
Figure 4.37. Running dataset graphic

Figure 4.38. Running dataset graphic-2









Linear Regression output of running dataset
Figure 4.39. Linear Regression output of running dataset

Decision Trees Regression output of running dataset
Figure 4.40. Decision Trees Regression output of running dataset









Random Forest Regression output of running dataset

Figure 4.41. Random Forest output of running dataset

Ridge Regression output of running dataset
Figure 4.42. Ridge Regression output of running dataset










CHAPTER 5.	CONCLUSION

In our study, we work on accelerometer, gyroscope and gps sensor data. We developed an application for Android software for data collecting. Data collection has been made in three different conditions which are sitting, walking and running. The main reason we collected data in three different situations was to observe the differences between them. Afterwards, we started to work on Python language in order to analyze sensor data. Exploratory data analysis has been made to three datasets with using pandas and pandas profiling libraries. Then, visualization works have been executed with matplotlib and seaborn libraries. Regression analysis has been performed to determine the relationship between accelerometer and gyroscope data. As regression method, linear regression, decision trees regression, random forest regression, and ridge regression have been used. Folium library has been used to show GPS data on a map. To bring those works together, a GUI has been developed with using Tkinter library.
Thanks to developed GUI, users can select the dataset(sitting, walking or running) and the regression method(Random Forest Regression, Decision Trees Regression, Linear Regression or Ridge Regression) or exploratory data analysis then, execute selected data analysis method on selected dataset.

Accelerometer and gyroscope data in sitting situation are more stable than walking and running situations. On the other hand, accelerometer and gyroscope data are the least stable in running situation. By making consistent assumptions according to the data change, we have determined the types of movement. In line with this determination, we have made a classification to be used in new applications to be developed.
 
REFERENCES



[1]
	Erdaş, Ç., Atasoy, I., Açıcı, K., Oğul, H., Integrating features for accelerometer-based activity recognition.

[2]
	http://news.mit.edu/2010/android-abelson-0819

[3]
	https://scikit-learn.org/stable/faq.html

[4]
	https://matplotlib.org/

[5]
	https://python-visualization.github.io/folium/

[6]
	https://medium.com/@gongster/dask-an-introduction-and-tutorial-b42f901bcff5

[7]
	https://realpython.com/linear-regression-in-python/

[8]
	https://medium.com/pursuitnotes/decision-tree-regression-in-6-steps-with-python-1a1c5aa2ee16
		
[9]		https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

[10]		https://towardsdatascience.com/understanding-random-forest-58381e0602d2
		
[11]		https://www.geeksforgeeks.org/optimization-tips-python-code/

 
ÖZGEÇMİŞ



Berk Doğuş was born in Saray/Tekirdağ on 27.05.1996. He completed his elementary, secondary and high school education here in 2014. He graduated from Sakarya University Information Systems Engineering department. At 2018 and 2019, he completed his business and software development internship at “PAS South East Europe San ve Tic. Limited Sti.”.

Kaan Orhun Kanal was born in İstanbul on 20.12.1996. He completed his primary school in Tantavi Primary school. Then, he went to Suat Terimer Anatolian High School for his high school education. He is still studying at Sakarya University Information Systems Engineering department.
 
ISE 402 BİTİRME ÇALIŞMASI
DEĞERLENDİRME VE SÖZLÜ SINAV TUTANAĞI

