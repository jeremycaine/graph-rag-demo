# watsonx.ai setup

https://dataplatform.cloud.ibm.com/wx/home?context=wx 

Projects > New Project > 
    - Name: `graph-rag`
    - Select object storage: `my-object-storage`

Project `graph-rag`
- Manage > Service Integrations > Associate Service
    - `Watson Machine Learning-1f3` (find the instance of machine learning engine)
        - Plan: Essentials
        - Location: Dallas

- Manage > General
    Project id: `[get the project id]` (WX_PROJECT_ID)
    
Get an API key (WX_API_KEY)
- IBM Cloud - Manage IAM > API Key