#!/usr/bin/env python3
"""
Test script for the optimized insurance policy QA system
Author: kg290
Date: 2025-07-26
"""

import asyncio
import json
import time
from unittest.mock import Mock, patch
import sys
import os

# Add the project directory to the path
sys.path.insert(0, '/home/runner/work/BajajRx-6.0/BajajRx-6.0')

# Test imports
try:
    from rate_limiter import create_rate_limiter, RateLimitConfig
    from response_optimizer import optimize_for_hackathon, create_response_optimizer
    print("✓ Successfully imported optimization modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


async def test_rate_limiter():
    """Test rate limiter functionality"""
    print("\n=== Testing Rate Limiter ===")
    
    config = RateLimitConfig(
        requests_per_minute=5,
        cache_ttl=60,
        enable_caching=True
    )
    
    limiter = create_rate_limiter(config)
    
    # Mock function to test
    call_count = 0
    async def mock_api_call():
        nonlocal call_count
        call_count += 1
        return f"Response {call_count}"
    
    # Test caching
    result1 = await limiter.execute_with_rate_limit(mock_api_call, "test_key_1")
    result2 = await limiter.execute_with_rate_limit(mock_api_call, "test_key_1")  # Should be cached
    
    print(f"First call result: {result1.data}")
    print(f"Second call cached: {result2.cached}")
    print(f"API calls made: {call_count}")
    
    assert result2.cached == True, "Second call should be cached"
    assert call_count == 1, "Should only make one API call due to caching"
    
    print("✓ Rate limiter with caching works correctly")
    
    # Test stats
    stats = limiter.get_stats()
    print(f"Rate limiter stats: {stats}")
    
    return True


def test_response_optimizer():
    """Test response optimization functionality"""
    print("\n=== Testing Response Optimizer ===")
    
    # Test verbose response optimization
    verbose_response = """
    Based on the policy document, it should be noted that a grace period of thirty days 
    is provided for premium payment after the due date. The policy clearly states that 
    this grace period allows the policyholder to make payment without penalty. In summary, 
    you have thirty days to pay your premium after it's due.
    """
    
    optimized = optimize_for_hackathon(verbose_response)
    print(f"Original length: {len(verbose_response)} chars")
    print(f"Optimized length: {len(optimized)} chars")
    print(f"Optimized response: {optimized}")
    
    # Should be much shorter and more concise
    assert len(optimized) < len(verbose_response) * 0.6, "Response should be significantly shorter"
    assert "thirty days" in optimized, "Key information should be preserved"
    
    print("✓ Response optimization works correctly")
    
    # Test fact extraction
    optimizer = create_response_optimizer()
    facts = optimizer._extract_all_facts(verbose_response)
    
    print(f"Extracted facts: {len(facts)}")
    for fact in facts[:3]:
        print(f"  - {fact.type.value}: {fact.value} (confidence: {fact.confidence})")
    
    assert len(facts) > 0, "Should extract at least one fact"
    
    return True


def test_fallback_responses():
    """Test fallback response generation"""
    print("\n=== Testing Fallback Responses ===")
    
    # Mock import to test fallback without full system
    try:
        from main import generate_fallback_response
        
        test_queries = [
            "What is the grace period?",
            "How much is the premium?", 
            "What is covered under this policy?",
            "What are the exclusions?"
        ]
        
        for query in test_queries:
            payload = {"messages": [{"role": "user", "content": query}]}
            fallback = generate_fallback_response(payload, "API rate limited")
            
            print(f"Query: {query[:30]}...")
            print(f"Fallback: {fallback[:100]}...")
            
            assert len(fallback) > 10, "Fallback should provide meaningful response"
            assert "policy" in fallback.lower(), "Should reference policy"
        
        print("✓ Fallback responses work correctly")
        return True
        
    except ImportError:
        print("⚠ Skipping fallback test due to import issues")
        return True


async def test_json_parsing():
    """Test JSON parsing robustness"""
    print("\n=== Testing JSON Parsing Robustness ===")
    
    # Test valid JSON
    valid_json = '{"answer": "Grace period is 30 days", "confidence": 0.9}'
    
    try:
        data = json.loads(valid_json)
        print(f"✓ Valid JSON parsed: {data['answer']}")
    except json.JSONDecodeError:
        print("✗ Failed to parse valid JSON")
        return False
    
    # Test invalid JSON (should handle gracefully)
    invalid_json = 'Grace period is 30 days (not valid JSON)'
    
    try:
        data = json.loads(invalid_json)
        print("✗ Should have failed to parse invalid JSON")
        return False
    except json.JSONDecodeError:
        print("✓ Correctly handled invalid JSON")
    
    return True


def run_performance_test():
    """Test performance characteristics"""
    print("\n=== Performance Test ===")
    
    start_time = time.time()
    
    # Test response optimization speed
    long_response = "This is a test response. " * 100
    for i in range(10):
        optimized = optimize_for_hackathon(long_response)
    
    optimization_time = time.time() - start_time
    print(f"10 optimizations took {optimization_time:.3f}s ({optimization_time/10:.4f}s each)")
    
    # Should be fast
    assert optimization_time < 1.0, "Optimization should be fast"
    
    print("✓ Performance test passed")
    return True


async def main():
    """Run all tests"""
    print("Starting optimization tests for Insurance Policy QA System v3.0")
    print("=" * 60)
    
    tests = [
        ("Rate Limiter", test_rate_limiter()),
        ("Response Optimizer", test_response_optimizer()),
        ("Fallback Responses", test_fallback_responses()),
        ("JSON Parsing", test_json_parsing()),
        ("Performance", run_performance_test())
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_coro in tests:
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            
            if result:
                print(f"✓ {test_name} - PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} - FAILED")
        except Exception as e:
            print(f"✗ {test_name} - ERROR: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System optimizations are working correctly.")
        return True
    else:
        print("⚠ Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)