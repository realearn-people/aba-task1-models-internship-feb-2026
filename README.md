# aba-task1-models-internship-feb-2026
This is a repository that contains the experiments for ABA Task 1, jointly conducted with internship students in February 2026. 

จะมี 3 folers หลักๆ :
1. dataset
2. model_code
3. model_result
และไฟล์รวมผลลัพธ์ของทุกโมเดล
- ตารางเทียบpred_sentiment_model

ใน folder
- dataset
  - จะมี 2 dataset
    - ABA Dataset (remove off) #dataset นี้จะไม่มีแถว (topic, sentiment ที่เป็น off)
    - Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT #dataset นี้จะยังมีข้อมูลที่เป็น off อยู่

- model_code
  - จะแบ่งออกเป็นทั้งหมด 5 folder ตามวิธีการทดลอง
    - model_code_ABA_T5
    - model_code_auto_finetune
      - ภายในจะมี code ที่ใช้รันของแต่ละ โมเดล (bert, bart, roberta, T5)
    - model_code_kfold
      - ภายในจะมี code ที่ใช้รันของแต่ละ โมเดล (bert, bart, roberta, T5)
    - model_code_remove_off
      - ภายในจะมี code ที่ใช้รันของแต่ละ โมเดล (bert, bart, roberta, T5)
    - model_code_with_off
      - ภายในจะมี code ที่ใช้รันของแต่ละ โมเดล (bert, bart, roberta, T5)

- model_result
  - จะแบ่งออกเป็นทั้งหมด 5 folder ตามวิธีการทดลอง
    - model_result_ABA_T5
    - model_result_auto_finetune
      - ภายในจะมี folers แยกผลลัพธ์ของแต่ละโมเดล (bert, bart, roberta, T5)
    - model_result_kfold
      - ภายในจะมี folers แยกผลลัพธ์ของแต่ละโมเดล (bert, bart, roberta, T5)
    - model_result_remove_off
      - ภายในจะมี folers แยกผลลัพธ์ของแต่ละโมเดล (bert, bart, roberta, T5)
    - model_result_with_off
      - ภายในจะมี folers แยกผลลัพธ์ของแต่ละโมเดล (bert, bart, roberta, T5)
