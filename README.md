# Procurement spend pipeline

The two build CSVs at the repository root are ready for review. All processing runs locally on CPU. Use Python 3.11 or 3.12. From this directory install dependencies once:

```sh
python -m pip install -r requirements.txt
```

Unzip the original task kit **outside** the repository. Pass the directory containing company_A and company_B. The full build command is:

```sh
python pipeline.py --kit ../Cantill_DataML_Task_Kit
```

Verify every output row and reproduce the hidden-label experiment:

```sh
python pipeline.py --config configs/A.json --data ../Cantill_DataML_Task_Kit/company_A --hide-labels --out private_eval/hidden.csv
python verify.py ../Cantill_DataML_Task_Kit --hidden private_eval/hidden.csv
```


## Blind run

Keep the original kit locally for training and taxonomy fallback. Put fresh rows in separate folders with the filenames from each existing config. Keep each company's monthly FX file alongside its fresh input if available. Use the original FX table there if omitted. Example commands:

```sh
python pipeline.py --config configs/A.json --data ../blind/A2 --reference ../Cantill_DataML_Task_Kit/company_A --out submission/Abhishek_Singh__A2.csv
python pipeline.py --config configs/B.json --data ../blind/B2 --reference ../Cantill_DataML_Task_Kit/company_B --out submission/Abhishek_Singh__B2.csv
python pipeline.py --config submission/C.json --data ../blind/C --out submission/Abhishek_Singh__C.csv
```

For C, copy configs/C.example.json to submission/C.json. Set filenames and column names from the actual blind files. List taxonomy columns in root-to-leaf order (one to four). For a net USD file, omit currency and rate mappings if absent. Remove any optional field whose column is absent. Set description and optional context to lists of columns. For gross amounts set tax to the rate column and tax_default to a fraction. Use decimal "," for decimal-comma amounts. Set date_formats to a list of Python strptime formats if needed. For localized statuses, approved accepts a list of normalized accepted values. A config can override the non_procurement regex. Keep category decisions automatic. Run from the repo root.

## Delivery

Publish the **contents of this project** to a public GitHub repository. Keep private inputs and local_audit outside version control. Submit the repo URL using the original email form before 10 AM IST on 19 November 2026. After the freeze, commit only the three named blind outputs and submission/C.json by noon IST. The example C config is a schema guide and needs the actual C headers.
