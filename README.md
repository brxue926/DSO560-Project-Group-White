**Project Goal and Group**

This repo is a collection of codes our group wrote for Text Classification Project of DSO560 class. Our client is Thread Together, an Australian non-profit organization. (https://www.threadtogether.org)

Professor: Yu Chen

Group White:

- Jiayue (Daniel) Chen (jiayuech@usc.edu)
- Xinyi (Alex) Guo (xinyiguo@usc.edu)
- Yuyao Shen (yuyaoshe@usc.edu)
- Nanchun (Aslan) Shi (nanchuns@usc.edu)
- Bingru Xue (bingruxu@usc.edu)

**Project Overview**

Thread Together has huge inventory of women clothing. Each product possesses several attributes such as occasion, color, etc. Within each attribute group, each product possesses single or multiple values, such as "workout" for occasion or "black" for color. While only a proportion of products in the current inventory of Thread Together are tagged with these values, our client is looking for predictive models such that when given some information about a product (e.g. description, brand, etc.), the models are able to predict the attribute value(s). This will reduce labor cost and increase efficiency. 

We are required to pick 5 attributes groups, and build models for each of them. As results, when we input product information, the models will output predicted values for each of the attribute. 

The 5 attributes we chose are:

- Occasion (as required)
- Style (as required)
- Category
- Prints/Pattern
- Fit

We were provided a subset of all inventory data, and also a tagged list of products made by domain experts. With these data, we are able to train the models and make prediction. In the repo one could find codes from each of our group member, as each of us is responsible for one attribute. There is also a aggregation of all codes (in COMBINED MODEL folder) so that our client could use directly. There will not be a summary report as we are not required to, but there is a brief summary of the architectures for each model and logic process.

If you have any questions or further interests, you could contact us.
