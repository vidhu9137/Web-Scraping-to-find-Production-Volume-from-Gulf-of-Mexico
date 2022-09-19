# Web-Scraping-to-find-Production-Volume-from-Gulf-of-Mexico
Well-production data for 2019 and 2020 in the Gulf of Mexico is taken out in a clean output file using Web Scraping

OBJECTIVE: Python script that upon execution (from an arbitrary location) navigates to the BSEE website, checks for latest updates on certain datasets, downloads only necessary files, and generates/refreshes a clean output file.

STEPS FOLLOWED: 

1. Script first downloads zipped well-production data for 2019 and 2020 in the Gulf of Mexico from this website:
https://www.data.bsee.gov/Main/OGOR-A.aspx

2. Script writes to a sub-folder (“Download”) of the working directory where the python script is executed.

3. Unzipped file is encoded to show last updated timestamp like ogora2019_20220601.

4. Downloaded files are converted to clean format  “production.csv” showing necessary columns like lease_number; production_date, product_code, monthly oil
and gas volumes, operator_num. 

NOTE: Script is written with these benefits: 
      It ONLY downloads and processes a particular file if the “last updated” flag on the website suggest the data has been refreshed. 
      Script has generic codes: It can download data of other period just by changing a single line of code. Also, upon light modification, it can pick up all files even if it's run a few years from now?

