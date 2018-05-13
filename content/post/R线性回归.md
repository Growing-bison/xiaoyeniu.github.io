---
title: "数据处理-简单模型与绘图-01"
Description: "作为和师弟师妹简单介绍的文章，主要是有关线性回归建模与图形绘制"
date: 2018-05-10T16:16:59+08:00
draft: True
tags: ["R","线性回归","绘图"]
share: true
---

<!--more-->

================

### 首先加载R的包

``` r
library(flexdashboard)
library(rJava)
library(car)
```

    ## Loading required package: carData

``` r
library(plotly)
```

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'plotly'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     last_plot

    ## The following object is masked from 'package:stats':
    ## 
    ##     filter

    ## The following object is masked from 'package:graphics':
    ## 
    ##     layout

``` r
library(ggplot2)
```

### 1. 散点图绘制

首先进行基本数据的观察。这里主要几个函数 - 设置文件路径： &gt; setwd()

-   读入csv文件 &gt; read.csv

-   ggplot2函数 &gt; ggplot()

``` r
setwd("C:/Users/Administrator/Desktop/for_404-figure/datasource")
location1_data = read.csv('location1_ndvi1.csv',encoding='UTF-8')
location1_lm = lm(location1_data$dry_matter.g.m2~location1_data$log_ndvi)
ggplot(location1_data,
       aes(x=location1_data$log_ndvi,
           y=location1_data$dry_matter.g.m2.,
           label=rownames(location1_data[1])))+
  geom_point(size=2,color="blue")+
  geom_text(hlabel=rownames(location1_data[1]),
             alpha=0.3,hjust=0,vjust=2,size=2)+
  geom_rug()+
  xlab('log_ndvi')+
  ylab('dry matter:g/m2')
```

![](/img/figure_m3/unnamed-chunk-2-1.png)

``` r
summary(location1_lm)
```

    ## 
    ## Call:
    ## lm(formula = location1_data$dry_matter.g.m2 ~ location1_data$log_ndvi)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -73.25 -31.98 -10.31  23.62 212.98 
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)               316.34      37.07   8.534 3.30e-12 ***
    ## location1_data$log_ndvi   257.71      43.78   5.887 1.51e-07 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 51.45 on 65 degrees of freedom
    ##   (1 observation deleted due to missingness)
    ## Multiple R-squared:  0.3478, Adjusted R-squared:  0.3377 
    ## F-statistic: 34.66 on 1 and 65 DF,  p-value: 1.512e-07

### 2. 第一次 线性拟合结果中判断异常点

对第一次拟合模型不满意，希望探寻原数据中是否有异常值影响。
主要通过学生化残差——离群点、库克距离——强影响点、帽子值——高杠杆点

-   综合绘图函数:influencePlot() \# car包

``` r
#car::influencePlot(location1_lm, #id.method=list("x","y"),labels=rownames(location1_data[9]),id.n=nrow(location1_data[9]))
car::influencePlot(location1_lm, method="identify")
```

![](/img/figure_m3/unnamed-chunk-3-1.png)

    ##     StudRes        Hat      CookD
    ## 20 2.996216 0.14830962 0.69619267
    ## 45 4.941031 0.04515861 0.42443172
    ## 47 0.950764 0.12910081 0.06709942

### 3. 二次拟合：删除异常点后拟合

剔除部分异常点，重新拟合。

-   anova对比：根据P值是否显著，判断是否有差别。
    &gt; anova()

-   赤池信息准则:选择小值 &gt; AIC()

``` r
location1_data_edit20_3 = location1_data[-c(1,2,3,20,27,28,29,30,31,32,34,33,42,43,44,45,46),]
location1_lm2 = lm(location1_data_edit20_3$dry_matter.g.m2.~location1_data_edit20_3$log_ndvi)
summary(location1_lm2)
```

    ## 
    ## Call:
    ## lm(formula = location1_data_edit20_3$dry_matter.g.m2. ~ location1_data_edit20_3$log_ndvi)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -65.462 -21.262  -3.482  25.634 107.046 
    ## 
    ## Coefficients:
    ##                                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                        221.96      35.88   6.186  1.3e-07 ***
    ## location1_data_edit20_3$log_ndvi   156.39      41.68   3.753 0.000472 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 33.77 on 48 degrees of freedom
    ##   (1 observation deleted due to missingness)
    ## Multiple R-squared:  0.2268, Adjusted R-squared:  0.2107 
    ## F-statistic: 14.08 on 1 and 48 DF,  p-value: 0.0004719

``` r
car::influencePlot(location1_lm2, method="identify")
```

![](/img/figure_m3/unnamed-chunk-4-1.png)

    ##       StudRes        Hat      CookD
    ## 1   1.8470285 0.07352792 0.12889873
    ## 23  3.5837724 0.02433724 0.12848349
    ## 28  0.4995480 0.13561951 0.01988773
    ## 30  0.5140136 0.23810427 0.04192757
    ## 32 -2.0203538 0.02004473 0.03922777

``` r
# print('———————————————————————————————————————————————————————————————————————————')
# 对比两个模型效果——ANOVA
print(anova(location1_lm, location1_lm2))
```

    ## Analysis of Variance Table
    ## 
    ## Response: location1_data$dry_matter.g.m2
    ##                         Df Sum Sq Mean Sq F value    Pr(>F)    
    ## location1_data$log_ndvi  1  91732   91732  34.659 1.512e-07 ***
    ## Residuals               65 172036    2647                      
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# print('———————————————————————————————————————————————————————————————————————————')
# 对比两个模型效果——AIC
print(AIC(location1_lm, location1_lm2))
```

    ##               df      AIC
    ## location1_lm   3 722.1392
    ## location1_lm2  3 497.7956

### 4. 二次拟合结果：对比实测值

通过图形观察拟合效果，对比观察与预测值，绘制对比图。

-   构建dataframe的函数 &gt; data.frame()

-   绘图函数 &gt; plot() text() point()

-   颜色获取函数 &gt; hsv() hsv(h = 1, s = 1, v = 1, alpha=0.5)

颜色选取用hsv函数，对应hsv颜色空间参考https://codebeautify.org/hsv-to-rgb-converter

<http://sape.inf.usi.ch/quick-reference/ggplot2/colour>

[r color cheatsheet](https://www.nceas.ucsb.edu/~frazier/RSpatialGuides/colorPaletteCheatsheet.pdf)

``` r
fitdata_location1=data.frame(location1_data_edit20_3$dry_matter.g.m2.,
                             location1_data_edit20_3$log_ndvi)

fit_location1=predict(location1_lm2,
                      fitdata_location1,
                      type = "response")

fitdata_location1_end=data.frame(fitdata_location1,
                                 fit_location1)


plot(fitdata_location1_end$location1_data_edit20_3.log_ndvi,
     fitdata_location1_end$location1_data_edit20_3.dry_matter.g.m2.,
     xlab="ndvi",
     ylab = "dry matter:g/m2", 
     main = "location1_ndvi与产量拟合_")

text1_color = hsv(h = 1, s = 0, v = 0.7, alpha=0.5)  # 黑色

text(x=fitdata_location1_end$location1_data_edit20_3.log_ndvi,
     y=fitdata_location1_end$location1_data_edit20_3.dry_matter.g.m2.,
     lab=rownames(location1_data[1]),
     adj=c(0,-0.5),
     pch=0.5,
     col = text1_color, 
     cex = 0.6,  # 字体大小设置
     pos=2 # 位置
     )

points(fitdata_location1_end$location1_data_edit20_3.log_ndvi,
       fitdata_location1_end$fit_location1,
       pch=10,
       col=2  # 对应黑色
       )

text2_color = hsv(h = 1, s = 1, v = 1, alpha=0.3)  # 红色

text(x=fitdata_location1_end$location1_data_edit20_3.log_ndvi,
     y=fitdata_location1_end$fit_location1, 
     lab=rownames(location1_data[1]),
     adj=c(0,-0.5),
     pch=0.5,
     col=text2_color,
     cex = 0.6,
     pos =3,
     offset = 0.5  # 相对偏置距离
     )

legend("topleft",c("实际","拟合"), 
       pch = c(1,10),
       col = c(1,2),
       title = "location1_ndvi与产量拟合")
```

![](/img/figure_m3/unnamed-chunk-5-1.png)

``` r
summary(location1_lm2)
```

    ## 
    ## Call:
    ## lm(formula = location1_data_edit20_3$dry_matter.g.m2. ~ location1_data_edit20_3$log_ndvi)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -65.462 -21.262  -3.482  25.634 107.046 
    ## 
    ## Coefficients:
    ##                                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                        221.96      35.88   6.186  1.3e-07 ***
    ## location1_data_edit20_3$log_ndvi   156.39      41.68   3.753 0.000472 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 33.77 on 48 degrees of freedom
    ##   (1 observation deleted due to missingness)
    ## Multiple R-squared:  0.2268, Adjusted R-squared:  0.2107 
    ## F-statistic: 14.08 on 1 and 48 DF,  p-value: 0.0004719

### 5. 多个曲线对比

对多个月份的数据，绘制拟合曲线图，并在一张图中呈现。 - 求对数：log() - 合并两个表格（dataframe）:melt() - 绘图函数：qplot() - x轴的刻度间隔设置：scale\_x\_continuous()

``` r
library(ggplot2)
library(data.table)

#对比之前几个月的数据，即拟合结果需要保证前后一致性。06月、07月、08月，以及在相应位置的变化应该是呈现一种上升的关系
# test_x = rnorm(100,mean = 0, sd=1)  # 生成满足正态分布的随机数

test_x = log(runif(100,min = 0,max = 1))

test_x = as.data.frame(test_x)

a_06month = 80
b_06month = 220
a_07month = 55
b_07month = 185
a_o8month_morepoint = 160
b_o8month_morepoint = 210
a_o8month_lesspoint = 270
b_o8month_lesspoint = 320

df_data = data.frame(x=exp(test_x),
                     y_06=test_x[,1]*a_06month+b_06month,
                     y_07=test_x[,1]*a_07month+b_07month,
                     y_08_more=test_x[,1]*a_o8month_morepoint+b_o8month_morepoint,
                     y_08_less=test_x[,1]*a_o8month_lesspoint+b_o8month_lesspoint)

df_melt_data = melt(df_data, id=1:5, measure=2:5)

p = qplot(data = df_melt_data,
          x=df_melt_data$test_x,
          y=df_melt_data$value,geom='line',
          color=df_melt_data$variable,
          main = '拟合结果',
          xlab = 'NDVI value', 
          ylab = '模拟的产量：g/m2') 

plot_end = p + scale_x_continuous(breaks = seq(0,1,0.1),labels = seq(0,1,0.1)) + scale_color_hue('图例')

print(plot_end)
```

![](/img/figure_m3/unnamed-chunk-6-1.png)
