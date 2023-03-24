# Working with tests: locally and in CI

**Content:**
   * [Running tests locally](#tests-locally)
   * [Running tests in CI](#tests-in-ci)
   * [CI stages](#ci-stages)
   * [FAQ](#faq)

## <a name="tests-locally"></a>Running tests locally

Before pushing your changes to a remote fork, you will want to check that your code is working correctly. 
To do this, you can run tests locally.

> **HINT:** If you extract articles URLs from dynamic site, 
> make sure you use `selenium.webdriver.Chrome` and have `headless mode` enabled.

To run tests locally, you need to perform several steps in PyCharm:

1. Install tests dependencies (ensure you have activated your environment if you have such by running
   `.\venv\Scripts\activate`):
   ```bash
   python -m pip install -r requirements_qa.txt
   ```

2. Create a new configuration:


   ![](../images/tests/pycharm_create_configuration.png)

3. Choose `pytest` as a target:


   ![](../images/tests/pycharm_choose_pytest_template.png)

4. Fill `pytest` configuration and click `OK`:


   ![](../images/tests/pycharm_fill_pytest_configuration.png)

5. Run `pytest` configuration:


   ![](../images/tests/pycharm_run_pytest.png)
   
   This should run all the tests in the repository. You can inspect them by clicking through a list
   at the bottom of a screen.

   ![](../images/tests/pycharm_tests_report.png)

6. As you have some tests failing, you want to debug them. Then, first, you need to limit
   a scope of running tests and the mark level you want to get for an assignment. For example,
   you might want to run checks for a crawler configuration. Then you need to return 
   to configuration menu and pass additional parameters, like `-m stage_2_1_crawler_config_check`.
   

   ![](../images/tests/pycharm_control_tests_scope.png)
   
   You can choose any of the labels that are described in [`../pyproject.toml`](../../pyproject.toml)
   and combine with a mark. For example, running the aforementioned check for configuration for a
   mark 8 will look like `-m "mark8 and stage_2_1_crawler_config_check"`. 

> **HINT:** To running all tests for first assignment for mark 8: 
> `-m "mark8 and (stage_2_1_crawler_config_check or stage_2_2_crawler_check or stage_2_3_HTML_parser_check or stage_2_4_dataset_volume_check or stage_2_5_dataset_validation)"`

> **HINT:** When you want to debug a test, instead of running them, put a breakpoint at the potentially vulnerable
> place of code and execute debugging by clicking a 'bug' button.


## <a name="tests-in-ci"></a>Running tests in CI

Tests will never run until you create a Pull Request.   

The very first check happens 
exactly when you create a Pull Request. After that, each time you push changes in your fork,
CI check will be automatically started, normally within a minute or two. To see the results,
navigate to your PR and click either the particular step in the report at the end of a page,
or click **Checks** in the toolbar. 

![](../images/tests/ci_report.png)

![](../images/tests/ci_tab.png)

Inspect each step by clicking through the list to the left.

## <a name="ci-stages"></a>CI stages

1. Stage 1. Style
   1. Stage 1.1. PR Name
   1. Stage 1.2. Code style (`pylint`, `flake8`)
   
1. Stage 2. Crawler
   1. Stage 2.1. Crawler config validation (we ensure that crawler has certain sanity checks)
   1. Stage 2.2. `Crawler` instantiation validation
   1. Stage 2.3. `Parser` instantiation validation
   1. Stage 2.4. Articles downloading
   1. Stage 2.5. Dataset volume validation
   1. Stage 2.6. Dataset structure validation
   
1. Stage 3. Text Processing Pipeline
   1. Stage 3.1. Dataset sanity checks (we ensure that pipeline has certain sanity checks)
   1. Stage 3.2. `CorpusManager` sanity checks (we ensure that pipeline identifies all articles correctly)
   1. Stage 3.3. `MorphologicalToken` sanity checks (we ensure that pipeline displays all tokens appropriately)
   1. Stage 3.4. Admin data processing
   1. Stage 3.5. Student dataset processing
   1. Stage 3.6. Student dataset validation
   
1. Stage 4. Additional tasks
   1. stage 4.1. `POSFrequencyPipeline` checks
   1. Stage 4.2. Frequency visualization

## <a name="faq"></a>FAQ

If you still have questions about running tests, or you have problems with it,
we hope you will find a solution in [FAQ: Running tests](faq.md#faq-tests).
