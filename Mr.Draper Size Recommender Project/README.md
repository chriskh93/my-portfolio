## Mr.Draper Size Recommender Project

Mr.Draper, the first and fasting growing digital styling service for men in the Middle East. It works as follows:
* A new user joins through the app and fills a style quiz
* In the style quiz, the user enters their clothing related information such as weight, height, known waist size, and known shirt size
* The user then proceeds to request a box
* A stylist at Mr.Draper fills the box based on the information the user inputted on the style quiz
* The box is delivered to the user
* The user chooses as many items from the box and sends it back
* The user is charged for the items they took

Mr.Draper is facing a problem. The problem is that their stylists are spending too much time trying to figure out the pieces of clothing that differ in size with brand. 

For example: I may be a size S with a T-shirt from Ralph Lauren, and a size M with a T-shirt from Scotch & Soda.

This in turn is causing size-related returns of items, and therefore costing Mr.Draper money.

The task here was to develop a Size Recommender that takes user inputted data from the style quiz and returns a recommended size for each product, thus allowing the stylist to give the users more accurate sizes in no time.

The files in this project are as follows:
* [Size Recommender Readme.pdf](https://github.com/chriskh93/my-portfolio/blob/main/Mr.Draper%20Size%20Recommender%20Project/Size%20Recommender%20Readme.pdf): An instruction manual on how to use the Size Recommender
* [api.py](https://github.com/chriskh93/my-portfolio/blob/main/Mr.Draper%20Size%20Recommender%20Project/api.py): The Python code that enables the API integration of the model onto the server
* [user_products_271120.csv](https://github.com/chriskh93/my-portfolio/blob/main/Mr.Draper%20Size%20Recommender%20Project/user_products_271120.csv): A provided dataframe that shows the different product_group_ids sent to the different users
* [users_271120.csv](https://github.com/chriskh93/my-portfolio/blob/main/Mr.Draper%20Size%20Recommender%20Project/users_271120.csv): A provided dataframe that shows the users details
* [product_group_sizes.csv](https://github.com/chriskh93/my-portfolio/blob/main/Mr.Draper%20Size%20Recommender%20Project/product_group_sizes.csv): A provided dataframe that shows the different sizes each product_group_id can take
* [product_groups_271120.csv](https://github.com/chriskh93/my-portfolio/blob/main/Mr.Draper%20Size%20Recommender%20Project/product_groups_271120.csv): A provided dataframe that shows the details of each product_group_id
* clf_bc_bn.joblib: A logsitic regression based model that takes a numerical "Bottom" size, person's weight, and a person's height, and returns a categorical "Bottom" size (Ex: Small, Medium, Large,etc.). "Bottom" in this case refers to pants and shorts.
* reg_OLS_st_ct.joblib: A  regression based model that takes a categorical "Top" size (Ex: Small, Medium, Large,etc.), person's weight, and a person's height, and returns a numerical "Top" size . "Top" in this case refers to tops.
* df_ar_3.csv: A developed dataframe used to record each users sizes across the different types, brands, categories, and fits of clothing
* df_btm_3.csv: A developed dataframe to record the current users known sizes across the different types, brands, categories, and fits of clothing
* [requirements.txt](https://github.com/chriskh93/my-portfolio/blob/main/Mr.Draper%20Size%20Recommender%20Project/requirements.txt): The Python requirements for a server to run our model
