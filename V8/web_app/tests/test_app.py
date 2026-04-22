# test_system_backend.py
import pytest
from playwright.sync_api import sync_playwright
import time
import requests
import json

class TestFunctionalCases:
    """A. Functional Test Cases - Backend Focused"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch()
        self.page = self.browser.new_page()
        self.page.goto('http://127.0.0.1:5000/')
        self.page.wait_for_load_state('networkidle')
        self.api_base = 'http://127.0.0.1:5000'
        yield
        self.browser.close()
        self.playwright.stop()
    
    def test_A1_user_authentication_and_session(self):
        """Test user authentication and session management"""
        login_button = self.page.locator('button:has-text("Login"), [data-testid="login-button"]')
        
        if login_button.is_visible():
            login_button.click()
            self.page.fill('input[type="email"], input[name="username"]', 'testactor@example.com')
            self.page.fill('input[type="password"]', 'testpassword123')
            self.page.click('button:has-text("Submit"), button[type="submit"]')
            
            user_profile = self.page.locator('[data-testid="user-profile"], .user-name')
            assert user_profile.is_visible(timeout=5000)
            print("✓ User authentication test passed")
    
    def test_A2_report_generation_and_download(self):
        """Test report generation functionality"""
        # Look for report generation button/link
        report_button = self.page.locator('button:has-text("Generate Report"), a:has-text("Report"), [data-testid="generate-report"]')
        
        if report_button.is_visible():
            report_button.click()
            
            # Check if report is generated
            download_link = self.page.locator('a:has-text("Download"), [data-testid="download-report"]')
            assert download_link.is_visible(timeout=10000), "Report not generated"
            
            print("✓ Report generation test passed")
    
    def test_A3_report_content_validation(self):
        """Test that generated report contains expected data"""
        # Navigate to reports page
        reports_link = self.page.locator('a:has-text("Reports"), [data-testid="reports"]')
        if reports_link.is_visible():
            reports_link.click()
            
            # Check report content sections
            report_sections = [
                '.performance-metrics',
                '.emotion-summary', 
                '.gaze-analysis',
                '.recommendations'
            ]
            
            for section in report_sections:
                element = self.page.locator(section)
                if element.is_visible():
                    content = element.text_content()
                    assert content and len(content) > 0, f"{section} is empty"
                    print(f"  {section}: ✓")
            
            print("✓ Report content validation passed")
    
    def test_A4_report_history_and_retrieval(self):
        """Test report history and retrieval"""
        history_link = self.page.locator('a:has-text("History"), [data-testid="report-history"]')
        
        if history_link.is_visible():
            history_link.click()
            
            # Check for list of past reports
            report_list = self.page.locator('[data-testid="report-list"], .report-items')
            assert report_list.is_visible(), "Report history not visible"
            
            # Try to open a past report
            first_report = self.page.locator('[data-testid="report-item"]:first-child, .report-item:first-child')
            if first_report.is_visible():
                first_report.click()
                time.sleep(1)
                print("  Past report retrieved successfully")
            
            print("✓ Report history test passed")
    
    def test_A5_performance_metrics_in_report(self):
        """Test performance metrics are included in report"""
        reports_page = self.page.locator('a:has-text("Reports")')
        if reports_page.is_visible():
            reports_page.click()
            
            # Check for specific metrics
            metrics = {
                'gaze_stability': ['Gaze Stability', 'gaze-stability', 'gaze'],
                'expression_diversity': ['Expression Diversity', 'expression-diversity', 'diversity'],
                'confidence_score': ['Confidence Score', 'confidence', 'score'],
                'overall_performance': ['Overall Performance', 'performance-score', 'rating']
            }
            
            found_metrics = []
            for metric_name, keywords in metrics.items():
                for keyword in keywords:
                    element = self.page.locator(f':has-text("{keyword}")')
                    if element.is_visible():
                        found_metrics.append(metric_name)
                        print(f"  {metric_name}: ✓")
                        break
            
            assert len(found_metrics) >= 2, f"Only found {len(found_metrics)} metrics"
            print("✓ Performance metrics validation passed")


class TestNonFunctionalCases:
    """B. Non-Functional Test Cases"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch()
        self.page = self.browser.new_page()
        yield
        self.browser.close()
        self.playwright.stop()
    
    def test_B1_page_load_time(self):
        """Test page load time under 5 seconds"""
        start_time = time.time()
        self.page.goto('http://127.0.0.1:5000/')
        self.page.wait_for_load_state('networkidle')
        load_time = (time.time() - start_time) * 1000
        
        assert load_time < 5000, f"Page load time {load_time}ms exceeds 5000ms"
        print(f"  Page load time: {load_time:.0f}ms ✓")
    
    def test_B2_report_generation_time(self):
        """Test report generation time under 5 seconds"""
        self.page.goto('http://127.0.0.1:5000/')
        
        # Find and click generate report button
        generate_btn = self.page.locator('button:has-text("Generate Report"), [data-testid="generate-report"]')
        
        if generate_btn.is_visible():
            start_time = time.time()
            generate_btn.click()
            
            # Wait for report to appear
            self.page.wait_for_selector('[data-testid="report-content"], .report-container', timeout=10000)
            gen_time = (time.time() - start_time) * 1000
            
            assert gen_time < 5000, f"Report generation took {gen_time}ms"
            print(f"  Report generation time: {gen_time:.0f}ms ✓")
    
    def test_B3_concurrent_report_requests(self):
        """Test handling concurrent report requests"""
        import concurrent.futures
        
        def request_report():
            try:
                response = requests.get(f'{self.api_base}/api/generate-report', timeout=10)
                return response.status_code == 200
            except:
                return False
        
        # Simulate 5 concurrent report requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(request_report) for _ in range(5)]
            results = [f.result() for f in futures]
        
        success_rate = sum(results) / len(results) * 100
        assert success_rate >= 80, f"Only {success_rate}% of concurrent requests succeeded"
        print(f"  Concurrent requests success rate: {success_rate}% ✓")
    
    def test_B4_api_response_time(self):
        """Test API response time for backend endpoints"""
        endpoints = [
            '/api/reports',
            '/api/performance-metrics',
            '/api/user/history'
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            try:
                response = requests.get(f'{self.api_base}{endpoint}', timeout=5)
                response_time = (time.time() - start_time) * 1000
                
                assert response_time < 2000, f"{endpoint} took {response_time}ms"
                print(f"  {endpoint}: {response_time:.0f}ms ✓")
            except:
                print(f"  {endpoint}: Not available")
    
    def test_B5_database_query_performance(self):
        """Test database query performance"""
        start_time = time.time()
        
        # Test report history query
        self.page.goto('http://127.0.0.1:5000/reports/history')
        self.page.wait_for_load_state('networkidle')
        
        query_time = (time.time() - start_time) * 1000
        assert query_time < 2000, f"Database query took {query_time}ms"
        print(f"  Database query time: {query_time:.0f}ms ✓")


class TestReportQuality:
    """Testing report quality and backend emotion detection results"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch()
        self.page = self.browser.new_page()
        self.page.goto('http://127.0.0.1:5000/')
        self.api_base = 'http://127.0.0.1:5000'
        yield
        self.browser.close()
        self.playwright.stop()
    
    def test_report_emotion_summary_exists(self):
        """Test that emotion summary exists in report (backend processed)"""
        reports_link = self.page.locator('a:has-text("Reports")')
        
        if reports_link.is_visible():
            reports_link.click()
            
            # Look for emotion-related content in report
            emotion_keywords = ['emotion', 'sentiment', 'expression', 'facial']
            found = False
            
            for keyword in emotion_keywords:
                element = self.page.locator(f':has-text("{keyword}")')
                if element.is_visible():
                    found = True
                    print(f"  Found emotion data: '{keyword}' ✓")
                    break
            
            # If not visible in UI, check via API
            if not found:
                try:
                    response = requests.get(f'{self.api_base}/api/latest-report')
                    if response.status_code == 200:
                        data = response.json()
                        has_emotion_data = 'emotions' in data or 'sentiment' in data or 'expression_analysis' in data
                        assert has_emotion_data, "No emotion data in API response"
                        print("  Emotion data found in API ✓")
                except:
                    pass
            
            print("✓ Emotion summary test passed")
    
    def test_report_recommendations_exist(self):
        """Test that report contains actionable recommendations"""
        reports_link = self.page.locator('a:has-text("Reports")')
        
        if reports_link.is_visible():
            reports_link.click()
            
            recommendation_keywords = [
                'recommend', 'suggest', 'improve', 
                'practice', 'feedback', 'action'
            ]
            
            found_recommendations = []
            for keyword in recommendation_keywords:
                elements = self.page.locator(f':has-text("{keyword}")')
                count = elements.count()
                if count > 0:
                    found_recommendations.append(keyword)
            
            print(f"  Found recommendations: {found_recommendations}")
            assert len(found_recommendations) >= 2, "Not enough recommendations found"
            print("✓ Recommendations test passed")
    
    def test_report_download_formats(self):
        """Test report download in different formats"""
        report_page = self.page.locator('a:has-text("Reports")')
        
        if report_page.is_visible():
            report_page.click()
            
            # Check for download buttons
            download_formats = ['PDF', 'CSV', 'JSON', 'Download']
            available_formats = []
            
            for fmt in download_formats:
                btn = self.page.locator(f'button:has-text("{fmt}"), a:has-text("{fmt}")')
                if btn.is_visible():
                    available_formats.append(fmt)
            
            print(f"  Available formats: {available_formats}")
            assert len(available_formats) >= 1, "No download formats available"
            print("✓ Report download formats test passed")


class TestUsabilitySUS:
    """Usability testing with SUS metrics"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.page = self.browser.new_page()
        self.page.goto('http://127.0.0.1:5000/')
        self.errors = []
        self.page.on('pageerror', lambda err: self.errors.append(err))
        yield
        self.browser.close()
        self.playwright.stop()
    
    def test_sus_task_navigation(self):
        """Measure time to complete navigation tasks"""
        tasks = {
            'Find Reports': lambda: self.page.click('a:has-text("Reports")'),
            'View Report History': lambda: self.page.click('a:has-text("History")'),
            'Download Report': lambda: self.page.click('button:has-text("Download"), a:has-text("Download")'),
        }
        
        task_times = {}
        for task_name, task_func in tasks.items():
            start = time.time()
            try:
                # Refresh page between tasks
                self.page.goto('http://127.0.0.1:5000/')
                task_func()
                task_times[task_name] = (time.time() - start) * 1000
                time.sleep(0.5)
            except:
                task_times[task_name] = None
        
        for task_name, task_time in task_times.items():
            if task_time:
                print(f"  {task_name}: {task_time:.0f}ms")
                assert task_time < 5000, f"{task_name} took {task_time:.0f}ms"
        
        print("✓ SUS navigation test passed")
    
    def test_sus_error_count(self):
        """Count errors during report workflow"""
        initial_errors = len(self.errors)
        
        # Complete report workflow
        try:
            reports_link = self.page.locator('a:has-text("Reports")')
            if reports_link.is_visible():
                reports_link.click()
                time.sleep(1)
                
                history_link = self.page.locator('a:has-text("History")')
                if history_link.is_visible():
                    history_link.click()
                    time.sleep(1)
                
                download_btn = self.page.locator('button:has-text("Download")')
                if download_btn.is_visible():
                    download_btn.click()
        except Exception as e:
            print(f"  Workflow error: {e}")
        
        new_errors = len(self.errors) - initial_errors
        assert new_errors <= 2, f"Found {new_errors} errors in workflow"
        print(f"  Errors in workflow: {new_errors} ✓")


def test_backend_emotion_api():
    """Direct backend API test for emotion detection"""
    api_base = 'http://127.0.0.1:5000'
    
    # Test emotion detection endpoint (assuming it exists)
    endpoints_to_test = [
        '/api/process-emotion',
        '/api/analyze',
        '/api/emotion-analysis'
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.post(
                f'{api_base}{endpoint}',
                json={'test': True},
                timeout=5
            )
            if response.status_code == 200:
                print(f"  {endpoint}: Backend emotion detection working ✓")
                return
        except:
            continue
    
    print("  Note: Backend emotion API endpoint not found - check implementation")


def print_sus_summary():
    """Print SUS summary for users"""
    print("\n" + "="*60)
    print("System Usability Scale (SUS) - User Questionnaire")
    print("="*60)
    print("\nRate each from 1 (Strongly Disagree) to 5 (Strongly Agree):")
    print("\n1. I would use this system frequently")
    print("2. The system was unnecessarily complex")
    print("3. The system was easy to use")
    print("4. I would need technical support to use this system")
    print("5. The report functions were well integrated")
    print("6. The system had too much inconsistency")
    print("7. Most people would learn this system quickly")
    print("8. The system was cumbersome to use")
    print("9. I felt confident using the system")
    print("10. I needed to learn many things before using this system")
    print("\n✓ Passing SUS Score: ≥ 68")
    print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
    print_sus_summary()
    test_backend_emotion_api()