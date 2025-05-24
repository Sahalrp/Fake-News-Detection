"""
News verification module using LLM and web search.
"""
from typing import Dict, List, Optional, Tuple, Any
import re
import time
import json
import hashlib
from datetime import datetime

from .ollama_client import OllamaClient
from .web_search import WebSearch
from .logger import logger, log_exception, log_data

class NewsVerifier:
    """
    Class for verifying news articles using LLM and web search.
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1:7b",
        temperature: float = 0.2,  # Reduced temperature for more consistent results
        max_tokens: int = 800,  # Reduced max tokens for faster responses
        fast_mode: bool = True  # New parameter for fast verification
    ):
        """
        Initialize the news verifier.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
            fast_mode: Whether to use fast verification mode (fewer claims, searches)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fast_mode = fast_mode

        self.ollama_client = OllamaClient()
        self.web_search = WebSearch()

        # Cache for verification results - using a class variable for persistence across instances
        if not hasattr(NewsVerifier, '_shared_cache'):
            NewsVerifier._shared_cache = {}
        self._cache = NewsVerifier._shared_cache

        # Enhanced system prompt for the LLM with more balanced instructions and up-to-date knowledge
        self.system_prompt = """
        You are an expert fact-checking assistant with extensive knowledge in journalism, history, politics, science, and current events.
        Your mission is to accurately assess news articles by verifying their factual claims using internet search results.
        You must be balanced and fair in your assessment, avoiding bias in either direction.

        CRITICAL: You must prioritize the MOST RECENT information from search results, especially for current events, elections, and breaking news.
        The search results contain the most up-to-date information available - trust this information over any prior knowledge you may have.

        IMPORTANT: Be extremely careful about labeling content as "fake" - only do so when there is clear evidence of fabrication or significant factual errors.
        When in doubt, lean toward classifying content as "real" or "mixed" rather than "fake."

        CORE PRINCIPLES:
        1. ACCURACY: Factual correctness is your highest priority
        2. CHARITY: Give articles the benefit of the doubt when evidence is limited
        3. RECENCY: Prioritize the most recent information from search results
        4. THOROUGHNESS: Examine every claim and detail meticulously
        5. BALANCE: Avoid bias toward classifying content as either real or fake
        6. EVIDENCE-BASED: Base all conclusions on verifiable evidence from search results
        7. NUANCE: Recognize that articles may contain a mix of accurate and inaccurate information
        8. CONTEXT: Consider the broader context and significance of any inaccuracies

        VERIFICATION METHODOLOGY:
        1. Identify all factual claims in the article (people, events, statistics, quotes, etc.)
        2. Cross-reference each claim with search results from reliable sources
        3. Verify the existence and accuracy of all named entities (people, organizations, places)
        4. Check if events described actually occurred as described
        5. Verify dates, numbers, statistics, and specific facts
        6. Assess if quotes are accurately attributed and reported
        7. Evaluate if the overall narrative is consistent with reliable reporting
        8. Consider if any inaccuracies appear to be honest mistakes rather than deliberate misinformation

        IMPORTANT CONSIDERATIONS:
        - Distinguish between major factual errors and minor inaccuracies
        - Consider the significance of any errors in the context of the entire article
        - Recognize that absence of evidence is not necessarily evidence of absence
        - Be aware that search results may be incomplete or limited
        - Consider the reliability and diversity of the sources in search results
        - Evaluate the percentage of claims that are supported vs. unsupported
        - Remember that news articles may simplify complex topics without being "fake"
        - Consider that different legitimate sources may present different perspectives on the same events

        VERDICT GUIDELINES:
        - REAL: The article's key claims are substantially supported by reliable sources, even if there are minor inaccuracies
        - MIXED: The article contains a mix of accurate and inaccurate information, with no clear intent to mislead
        - FAKE: Multiple significant factual errors are found AND key claims are contradicted by reliable sources, suggesting deliberate misinformation

        CONFIDENCE LEVELS:
        - HIGH: Clear evidence supporting your verdict from multiple reliable sources
        - MEDIUM: Sufficient evidence supporting your verdict, but with some limitations
        - LOW: Limited evidence available, making a definitive determination difficult

        SPECIAL INSTRUCTIONS:
        1. Verify if people, organizations, and events mentioned in the article exist and are accurately described
        2. Carefully weigh the significance of any factual errors - minor errors do not invalidate the entire article
        3. Pay close attention to the search results and prioritize them in your assessment
        4. Consider the overall accuracy percentage - what percentage of claims are supported by evidence?
        5. Clearly state your verdict as "VERDICT: REAL", "VERDICT: MIXED", or "VERDICT: FAKE" and confidence level as "CONFIDENCE: HIGH/MEDIUM/LOW"
        6. Include a percentage estimate of how much of the article appears to be factually accurate
        7. When search results are limited or inconclusive, default to a more charitable interpretation

        Remember: Your goal is to provide a balanced, evidence-based assessment that helps people understand the reliability of the information they're reading.
        Err on the side of caution before labeling content as fake news.
        """

    def extract_key_claims(self, article_text: str) -> List[str]:
        """
        Extract key claims from an article using the LLM.

        Args:
            article_text: The text of the news article

        Returns:
            List of key claims extracted from the article
        """
        logger.info("Extracting key claims from article")

        # Validate input
        if not article_text or not article_text.strip():
            logger.error("Empty article text provided")
            return []

        # Log article length
        article_length = len(article_text)
        logger.debug(f"Article length: {article_length} characters")

        # Log a sample of the article
        if article_length > 500:
            logger.debug(f"Article sample: {article_text[:500]}...")
        else:
            logger.debug(f"Article: {article_text}")

        # Create a more concise prompt for faster claim extraction
        prompt = f"""
        TASK: Extract 3 key factual claims from this article that can be verified.

        ARTICLE:
        {article_text}

        FACTUAL CLAIMS (list 3 specific, verifiable claims):
        1.
        """

        logger.debug("Sending claim extraction prompt to LLM")

        try:
            # Generate response from LLM with reduced tokens for faster response
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=500  # Reduced tokens for faster response
            )

            logger.debug(f"LLM response length: {len(response)} characters")

            # Log the response
            if len(response) > 500:
                logger.debug(f"LLM response (truncated): {response[:500]}...")
            else:
                logger.debug(f"LLM response: {response}")

            # Parse the response to extract claims
            claims = []

            # First try to extract numbered or bulleted claims
            logger.debug("Parsing response for structured claims")
            for line in response.split('\n'):
                # Look for numbered or bulleted claims
                if re.match(r'^[\d\-\*\•]+\.?\s', line.strip()):
                    claim = re.sub(r'^[\d\-\*\•]+\.?\s', '', line.strip())
                    if claim:
                        claims.append(claim)
                        logger.debug(f"Found structured claim: {claim}")

            # If no structured claims were found, try to split by newlines
            if not claims:
                logger.debug("No structured claims found, trying to extract by newlines")
                claims = [line.strip() for line in response.split('\n') if line.strip()]

                # Filter out lines that are likely not claims
                claims = [line for line in claims if len(line) > 10 and not line.startswith("Key Claims")]

                if claims:
                    logger.debug(f"Extracted {len(claims)} claims by newlines")
                else:
                    logger.warning("Failed to extract any claims from LLM response")

            # Limit to 5 claims
            claims = claims[:5]

            # Log the extracted claims
            logger.info(f"Extracted {len(claims)} key claims from article")
            for i, claim in enumerate(claims, 1):
                logger.debug(f"Claim {i}: {claim}")

            return claims

        except Exception as e:
            log_exception(e, "extract_key_claims")
            logger.error(f"Error extracting key claims: {e}")
            return []

    def verify_article(self, article_text: str) -> Dict[str, Any]:
        """
        Verify a news article using LLM and web search.

        Args:
            article_text: The text of the news article

        Returns:
            Dictionary with verification results
        """
        logger.info("Starting verification of article")

        try:
            # Validate input
            if not article_text or not article_text.strip():
                logger.error("Empty article text provided")
                raise ValueError("Empty article text provided")

            # Check cache first for identical articles - using a more reliable hash method
            article_text_normalized = ' '.join(article_text.strip().split())  # Normalize whitespace
            article_hash = hashlib.md5(article_text_normalized.encode('utf-8')).hexdigest()

            if article_hash in self._cache:
                logger.info("Using cached verification result for identical article")
                cached_result = self._cache[article_hash]
                # Add a timestamp to track when this result was retrieved from cache
                if "cache_retrieved_at" not in cached_result:
                    cached_result["cache_retrieved_at"] = datetime.now().isoformat()
                return cached_result

            # Log article length
            article_length = len(article_text)
            logger.info(f"Article length: {article_length} characters")

            # Step 1: Extract key claims
            logger.info("Step 1: Extracting key claims")
            claims = self.extract_key_claims(article_text)

            if not claims:
                logger.warning("No claims extracted, using default claim")
                claims = ["The article does not contain clear factual claims that can be verified."]

            # In fast mode, limit the number of claims to process
            if self.fast_mode and len(claims) > 3:
                logger.info(f"Fast mode: Limiting claims from {len(claims)} to 3")
                # Keep the first 3 claims as they're usually the most important
                claims = claims[:3]

            # Step 2: Search for information about each claim
            logger.info(f"Step 2: Searching for information about {len(claims)} claims")
            search_results = []

            for i, claim in enumerate(claims, 1):
                logger.info(f"Searching for claim {i}/{len(claims)}: {claim[:50]}...")

                try:
                    # Sanitize the claim
                    sanitized_claim = claim.strip()
                    logger.debug(f"Sanitized claim {i}: {sanitized_claim}")

                    # Search for information
                    logger.debug(f"Performing web search for claim {i}")
                    # Use more search results for better context, but still respect fast mode
                    num_results = 3 if self.fast_mode else 5
                    result = self.web_search.search_and_summarize(sanitized_claim, num_results=num_results)

                    # Log search result length
                    logger.debug(f"Search result length for claim {i}: {len(result)} characters")

                    # Add to results
                    search_result = {
                        "claim": sanitized_claim,
                        "search_results": result
                    }

                    search_results.append(search_result)
                    log_data(search_result, f"Search result for claim {i}")

                    # No delay between searches in fast mode
                    if not self.fast_mode:
                        logger.debug("Adding minimal delay between searches")
                        time.sleep(0.2)  # Reduced delay

                except Exception as e:
                    log_exception(e, f"verify_article - search for claim {i}")
                    logger.error(f"Error searching for claim '{claim}': {e}")

                    error_result = {
                        "claim": claim,
                        "search_results": f"Error performing search: {str(e)}"
                    }

                    search_results.append(error_result)
                    log_data(error_result, f"Error result for claim {i}")

            # Step 3: Ask the LLM to analyze the article and search results
            logger.info("Step 3: Analyzing article and search results with LLM")

            # Build an enhanced prompt with more detailed instructions
            prompt = f"""
            CRITICAL TASK: Analyze this news article and determine its factual accuracy based on the search results.
            IMPORTANT: Be balanced and charitable in your assessment - avoid being overly critical of minor inaccuracies.

            ARTICLE TEXT:
            {article_text}

            KEY CLAIMS AND SEARCH RESULTS:
            """

            for i, result in enumerate(search_results, 1):
                prompt += f"\n--- CLAIM {i}: {result['claim']} ---\n"
                prompt += f"SEARCH RESULTS:\n{result['search_results']}\n"

            prompt += """
            ANALYSIS INSTRUCTIONS:

            1. For each claim, carefully evaluate:
               - Is the claim substantially supported by reliable sources in the search results?
               - Are there significant contradictions between the claim and search results?
               - Is the claim exaggerated or misleading compared to what sources say?
               - If the claim cannot be verified, assume it may be true but unverifiable with current search results
               - Do the names, titles, and specific facts mentioned actually exist and are accurate?
               - Distinguish between major factual errors and minor inaccuracies or simplifications
               - Be charitable in your assessment - news articles often simplify complex topics

            2. CRITICAL: For claims about current events, elections, or recent news:
               - TRUST THE SEARCH RESULTS over any prior knowledge you may have
               - Pay special attention to dates mentioned in search results
               - Verify if events described have actually occurred according to the most recent information
               - Check if people hold the positions or titles claimed in the article
               - For election-related claims, verify the most current election results from search data
               - Consider that breaking news may have limited search coverage

            3. Consider these indicators when evaluating the article:
               - Claims contradicted by multiple reliable sources suggest potential inaccuracies
               - Information not found in search results is likely due to limited search coverage rather than fabrication
               - Quotes or statistics that slightly differ from original sources may be simplifications rather than misrepresentation
               - Only label people, organizations, or events as "non-existent" if you have strong evidence they don't exist
               - Minor errors in titles or positions likely indicate honest mistakes rather than deliberate misinformation
               - The overall percentage of claims that are supported vs. unsupported by evidence
               - Different legitimate sources may present different perspectives on the same events

            4. BALANCED ASSESSMENT: Evaluate what percentage of the article appears to be factually accurate.
               - Consider the significance of any errors in the context of the entire article
               - Weigh major factual errors more heavily than minor inaccuracies
               - Determine if the core message of the article is supported by evidence
               - Be charitable when evidence is limited - absence of evidence is not evidence of absence

            5. Provide your final assessment:
               - VERDICT: State "VERDICT: REAL" (for mostly accurate content), "VERDICT: MIXED" (for content with both accurate and inaccurate elements), or "VERDICT: FAKE" (only for content with clear fabrication or major errors) clearly
               - CONFIDENCE: State "CONFIDENCE: LOW/MEDIUM/HIGH" clearly
               - ACCURACY PERCENTAGE: Estimate what percentage of the article is factually accurate (e.g., "ACCURACY: 75%")
               - EXPLANATION: Provide detailed reasoning with specific references to the search results

            IMPORTANT: Only use "VERDICT: FAKE" when there is clear evidence of deliberate fabrication or multiple significant factual errors that fundamentally undermine the article's credibility. When in doubt, use "VERDICT: MIXED" instead.

            Remember: Your goal is to provide a balanced, evidence-based assessment. Consider the overall accuracy of the article rather than focusing solely on finding any error. Be charitable in your interpretation when evidence is limited.
            """

            logger.debug(f"Analysis prompt length: {len(prompt)} characters")
            logger.debug(f"Analysis prompt sample: {prompt[:500]}...")

            # Generate analysis with reduced tokens for faster response
            logger.info("Generating analysis with LLM")
            analysis = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_tokens=600  # Reduced tokens for faster response
            )

            # Log analysis
            logger.debug(f"Analysis length: {len(analysis)} characters")
            if len(analysis) > 500:
                logger.debug(f"Analysis sample: {analysis[:500]}...")
            else:
                logger.debug(f"Analysis: {analysis}")

            # Step 4: Extract the verdict and confidence
            logger.info("Step 4: Extracting verdict and confidence")
            verdict = "Unknown"
            confidence = "LOW"

            # Enhanced verdict extraction with more patterns and scoring
            analysis_upper = analysis.upper()
            logger.debug("Extracting verdict from analysis")

            # Define patterns with weights for scoring - more balanced approach with mixed category
            real_patterns = [
                ("VERDICT: REAL", 10),
                ("VERDICT IS REAL", 10),
                ("VERDICT: THE ARTICLE IS REAL", 10),
                ("THE ARTICLE IS REAL", 8),
                ("THE ARTICLE APPEARS TO BE REAL", 7),
                ("THE ARTICLE SEEMS REAL", 6),
                ("THE CLAIMS ARE SUPPORTED", 5),
                ("THE INFORMATION IS ACCURATE", 5),
                ("THE ARTICLE IS CREDIBLE", 5),
                ("THE ARTICLE IS TRUSTWORTHY", 5),
                ("THE ARTICLE IS LEGITIMATE", 5),
                ("FACTUALLY ACCURATE", 7),
                ("FACTUALLY CORRECT", 7),
                ("FACTS ARE CORRECT", 6),
                ("FACTS ARE ACCURATE", 6),
                ("CLAIMS ARE VERIFIED", 6),
                ("CLAIMS ARE SUPPORTED BY EVIDENCE", 7),
                ("SUPPORTED BY SOURCES", 6),
                ("CONSISTENT WITH SOURCES", 5),
                ("ACCURACY: 70%", 7),
                ("ACCURACY: 80%", 8),
                ("ACCURACY: 90%", 9),
                ("ACCURACY: 100%", 10),
                ("MOSTLY ACCURATE", 7),
                ("HIGHLY ACCURATE", 8),
                ("COMPLETELY ACCURATE", 9),
                ("SUBSTANTIALLY ACCURATE", 7),
                ("GENERALLY ACCURATE", 6),
                ("LARGELY ACCURATE", 6),
                ("BROADLY ACCURATE", 6)
            ]

            mixed_patterns = [
                ("VERDICT: MIXED", 10),
                ("VERDICT IS MIXED", 10),
                ("VERDICT: THE ARTICLE IS MIXED", 10),
                ("THE ARTICLE IS MIXED", 8),
                ("THE ARTICLE CONTAINS BOTH ACCURATE AND INACCURATE INFORMATION", 8),
                ("MIXED ACCURACY", 7),
                ("PARTIALLY ACCURATE", 6),
                ("SOME CLAIMS ARE ACCURATE", 5),
                ("SOME CLAIMS ARE INACCURATE", 5),
                ("ACCURACY: 40%", 6),
                ("ACCURACY: 50%", 7),
                ("ACCURACY: 60%", 6),
                ("PARTIALLY SUPPORTED", 5),
                ("SOME FACTUAL ERRORS", 5),
                ("CONTAINS SOME INACCURACIES", 5),
                ("CONTAINS BOTH ACCURATE AND INACCURATE INFORMATION", 7),
                ("SOMEWHAT MISLEADING", 5),
                ("SOMEWHAT ACCURATE", 5),
                ("MIXED RELIABILITY", 6),
                ("MIXED CREDIBILITY", 6)
            ]

            fake_patterns = [
                ("VERDICT: FAKE", 10),
                ("VERDICT IS FAKE", 10),
                ("VERDICT: THE ARTICLE IS FAKE", 10),
                ("THE ARTICLE IS FAKE", 8),
                ("THE ARTICLE IS FABRICATED", 9),
                ("THE ARTICLE IS FALSE", 8),
                ("DELIBERATELY MISLEADING", 8),
                ("INTENTIONALLY FALSE", 9),
                ("COMPLETELY FALSE", 9),
                ("ENTIRELY FABRICATED", 10),
                ("CLEAR FABRICATION", 9),
                ("MULTIPLE SIGNIFICANT FACTUAL ERRORS", 8),
                ("FUNDAMENTALLY INACCURATE", 8),
                ("ACCURACY: 0%", 10),
                ("ACCURACY: 10%", 9),
                ("ACCURACY: 20%", 8),
                ("ACCURACY: 30%", 7),
                ("COMPLETELY INACCURATE", 9),
                ("ENTIRELY FALSE", 9),
                ("NO FACTUAL BASIS", 8)
            ]

            # Score the analysis
            real_score = 0
            mixed_score = 0
            fake_score = 0

            # Log all pattern matches for debugging
            logger.debug("Checking for real news patterns:")
            for pattern, weight in real_patterns:
                if pattern in analysis_upper:
                    real_score += weight
                    logger.debug(f"  Found '{pattern}', adding {weight} to real score")

            logger.debug("Checking for mixed news patterns:")
            for pattern, weight in mixed_patterns:
                if pattern in analysis_upper:
                    mixed_score += weight
                    logger.debug(f"  Found '{pattern}', adding {weight} to mixed score")

            logger.debug("Checking for fake news patterns:")
            for pattern, weight in fake_patterns:
                if pattern in analysis_upper:
                    fake_score += weight
                    logger.debug(f"  Found '{pattern}', adding {weight} to fake score")

            logger.debug(f"Final scores - Real: {real_score}, Mixed: {mixed_score}, Fake: {fake_score}")

            # Calculate percentage of real vs fake based on scores
            total_score = real_score + mixed_score + fake_score

            # First, try to extract explicit accuracy percentage from the analysis
            accuracy_match = re.search(r'ACCURACY:\s*(\d+)%', analysis_upper)
            if accuracy_match:
                explicit_accuracy = int(accuracy_match.group(1))
                logger.info(f"Found explicit accuracy percentage: {explicit_accuracy}%")

                # Use the explicit accuracy if it's reasonable
                if 0 <= explicit_accuracy <= 100:
                    real_percentage = explicit_accuracy
                    fake_percentage = 100 - real_percentage
                    logger.info(f"Using explicit accuracy percentage: {real_percentage}% real")

                    # Set verdict based on the explicit accuracy
                    if real_percentage >= 70:
                        verdict = "Real"
                    elif real_percentage <= 30:
                        verdict = "Fake"
                    else:
                        verdict = "Mixed"  # New verdict category for borderline cases
                else:
                    logger.warning(f"Explicit accuracy percentage {explicit_accuracy}% is out of range, ignoring")
                    accuracy_match = None

            # If no explicit accuracy was found or it was invalid, use pattern matching scores
            if not accuracy_match and total_score > 0:
                # Calculate percentage of real (0-100)
                real_percentage = int((real_score / total_score) * 100)
                fake_percentage = 100 - real_percentage

                # Look for explicit verdict statements first
                explicit_verdict = None

                # Check for explicit real verdict
                for pattern in ["VERDICT: REAL", "VERDICT IS REAL", "THE ARTICLE IS REAL"]:
                    if pattern in analysis_upper:
                        explicit_verdict = "Real"
                        logger.info(f"Found explicit real verdict: '{pattern}'")
                        break

                # Check for explicit mixed verdict
                for pattern in ["VERDICT: MIXED", "VERDICT IS MIXED", "THE ARTICLE IS MIXED"]:
                    if pattern in analysis_upper and not explicit_verdict:
                        explicit_verdict = "Mixed"
                        logger.info(f"Found explicit mixed verdict: '{pattern}'")
                        break

                # Check for explicit fake verdict
                for pattern in ["VERDICT: FAKE", "VERDICT IS FAKE", "THE ARTICLE IS FAKE"]:
                    if pattern in analysis_upper and not explicit_verdict:
                        explicit_verdict = "Fake"
                        logger.info(f"Found explicit fake verdict: '{pattern}'")
                        break

                # Calculate percentages based on scores
                if total_score > 0:
                    # Calculate percentages for each category
                    real_percentage = int((real_score / total_score) * 70)  # Scale to leave room for mixed
                    mixed_percentage = int((mixed_score / total_score) * 20)
                    fake_percentage = 100 - real_percentage - mixed_percentage

                    # Adjust to ensure total is 100%
                    if real_percentage + fake_percentage != 100:
                        adjustment = 100 - (real_percentage + fake_percentage)
                        # Add adjustment to the largest percentage
                        if real_percentage >= fake_percentage:
                            real_percentage += adjustment
                        else:
                            fake_percentage += adjustment
                else:
                    # Default to mostly real if no scores (benefit of the doubt)
                    real_percentage = 70
                    fake_percentage = 30

                # If we have an explicit verdict, adjust the percentages to be more consistent
                if explicit_verdict == "Real" and real_percentage < 70:
                    real_percentage = max(real_percentage, 70)  # Ensure at least 70% real for explicit real verdict
                    fake_percentage = 100 - real_percentage
                    logger.info(f"Adjusted real percentage to {real_percentage}% based on explicit real verdict")
                elif explicit_verdict == "Mixed" and (real_percentage < 40 or real_percentage > 70):
                    real_percentage = 60  # Set to 60% real for mixed verdict
                    fake_percentage = 40
                    logger.info(f"Adjusted percentages to 60% real, 40% fake based on explicit mixed verdict")
                elif explicit_verdict == "Fake" and real_percentage > 30:
                    real_percentage = min(real_percentage, 30)  # Ensure at most 30% real for explicit fake verdict
                    fake_percentage = 100 - real_percentage
                    logger.info(f"Adjusted real percentage to {real_percentage}% based on explicit fake verdict")

                # Set the verdict based on scores and any explicit verdict
                if explicit_verdict:
                    verdict = explicit_verdict
                    logger.info(f"Using explicit verdict: {verdict}")
                elif real_score > (mixed_score + fake_score):
                    verdict = "Real"
                    logger.info(f"Determined verdict: Real (scores - Real: {real_score}, Mixed: {mixed_score}, Fake: {fake_score})")
                elif fake_score > (real_score + mixed_score):
                    verdict = "Fake"
                    logger.info(f"Determined verdict: Fake (scores - Real: {real_score}, Mixed: {mixed_score}, Fake: {fake_score})")
                elif mixed_score > 0:
                    verdict = "Mixed"
                    logger.info(f"Determined verdict: Mixed (scores - Real: {real_score}, Mixed: {mixed_score}, Fake: {fake_score})")
                elif real_score >= fake_score:
                    verdict = "Real"  # Bias toward real when in doubt
                    logger.info(f"Scores close, defaulting to Real (scores - Real: {real_score}, Mixed: {mixed_score}, Fake: {fake_score})")
                else:
                    verdict = "Mixed"
                    logger.info(f"Scores unclear, defaulting to Mixed (scores - Real: {real_score}, Mixed: {mixed_score}, Fake: {fake_score})")
            elif not accuracy_match:
                # If no scores and no explicit accuracy, default to mostly real (benefit of the doubt)
                real_percentage = 70
                fake_percentage = 30
                verdict = "Real"  # Default to real when we have no clear evidence
                logger.info(f"No scores or explicit accuracy found, defaulting to 70% real, 30% fake (benefit of the doubt)")

            # Enhanced confidence extraction
            logger.debug("Extracting confidence from analysis")

            confidence_patterns = [
                # High confidence patterns
                ("CONFIDENCE: HIGH", "High"),
                ("HIGH CONFIDENCE", "High"),
                ("CONFIDENCE LEVEL: HIGH", "High"),
                ("CONFIDENCE LEVEL IS HIGH", "High"),
                ("I AM HIGHLY CONFIDENT", "High"),
                ("WITH HIGH CONFIDENCE", "High"),

                # Medium confidence patterns
                ("CONFIDENCE: MEDIUM", "Medium"),
                ("MEDIUM CONFIDENCE", "Medium"),
                ("CONFIDENCE LEVEL: MEDIUM", "Medium"),
                ("CONFIDENCE LEVEL IS MEDIUM", "Medium"),
                ("MODERATE CONFIDENCE", "Medium"),
                ("WITH MEDIUM CONFIDENCE", "Medium"),

                # Low confidence patterns
                ("CONFIDENCE: LOW", "Low"),
                ("LOW CONFIDENCE", "Low"),
                ("CONFIDENCE LEVEL: LOW", "Low"),
                ("CONFIDENCE LEVEL IS LOW", "Low"),
                ("WITH LOW CONFIDENCE", "Low")
            ]

            for pattern, result in confidence_patterns:
                if pattern in analysis_upper:
                    # Standardize confidence to uppercase for consistency
                    confidence = result.upper()
                    logger.debug(f"Found confidence pattern '{pattern}', setting confidence to '{result.upper()}'")
                    break

            logger.info(f"Final verdict: {verdict}, confidence: {confidence}")

            # Prepare the result
            result = {
                "claims": claims,
                "search_results": search_results,
                "analysis": analysis,
                "verdict": verdict,
                "confidence": confidence,
                "real_percentage": real_percentage,
                "fake_percentage": fake_percentage,
                "model_name": self.model_name,
                "cache_created_at": datetime.now().isoformat(),
                "article_hash": article_hash  # Store the hash for reference
            }

            log_data(result, "Verification result")
            logger.info("Article verification completed successfully")

            # Cache the result for future use - ensure we're using the shared cache
            if not hasattr(NewsVerifier, '_shared_cache'):
                NewsVerifier._shared_cache = {}
            NewsVerifier._shared_cache[article_hash] = result
            self._cache = NewsVerifier._shared_cache

            return result

        except Exception as e:
            # Log the error
            log_exception(e, "verify_article")
            logger.error(f"Error in verify_article: {e}")

            # Return a structured error response
            error_message = f"Error: {str(e)}"

            error_result = {
                "claims": [],
                "search_results": [],
                "analysis": error_message,
                "verdict": "Unknown",
                "confidence": "LOW",
                "real_percentage": 50,
                "fake_percentage": 50,
                "error": error_message
            }

            log_data(error_result, "Verification error result")
            return error_result

    def check_factual_accuracy(self, article_text: str) -> Dict[str, Any]:
        """
        Specifically check for factual errors in an article by performing targeted web searches
        on key entities and claims.

        Args:
            article_text: The text of the news article

        Returns:
            Dictionary with factual accuracy results
        """
        logger.info("Checking factual accuracy of article")

        try:
            # Step 1: Extract named entities and key facts
            logger.info("Step 1: Extracting named entities and key facts")
            entity_prompt = f"""
            TASK: Extract ALL named entities and specific factual claims from this news article.

            ARTICLE TEXT:
            {article_text}

            INSTRUCTIONS:
            1. Identify ALL of the following from the article:
               - People (full names and titles/positions)
               - Organizations
               - Locations
               - Dates and time periods
               - Numerical claims and statistics
               - Specific events described
               - Direct quotes and who said them

            2. Format your response as a numbered list of entities/claims:
               1. [Entity/Claim 1]
               2. [Entity/Claim 2]
               etc.

            3. Be thorough and specific - include EVERY named entity and factual claim.
            4. Do not include general statements, opinions, or vague claims.
            5. For people, include their full name AND title/position as mentioned in the article.

            RESPONSE FORMAT:
            NAMED ENTITIES AND FACTUAL CLAIMS:
            1.
            2.
            etc.
            """

            # Extract entities and facts
            logger.info("Extracting named entities and factual claims")
            entities_response = self.ollama_client.generate(
                model=self.model_name,
                prompt=entity_prompt,
                temperature=0.1,  # Lower temperature for more focused extraction
                max_tokens=1000
            )

            # Parse the entities
            entities = []
            for line in entities_response.split('\n'):
                # Look for numbered lines
                if re.match(r'^\d+\.', line.strip()):
                    entity = re.sub(r'^\d+\.\s*', '', line.strip())
                    if entity and len(entity) > 3:  # Minimum length check
                        entities.append(entity)

            logger.info(f"Extracted {len(entities)} named entities and factual claims")

            # Check if this is an election-related article
            election_keywords = ["election", "president", "vote", "elected", "won", "winner", "2024", "inauguration"]
            is_election_related = any(keyword in article_text.lower() for keyword in election_keywords)

            # Add special entities for election-related articles
            if is_election_related:
                logger.info("Election-related article detected, adding special verification queries")
                if "trump" in article_text.lower():
                    entities.append("Donald Trump 2024 election results")
                if "biden" in article_text.lower():
                    entities.append("Joe Biden 2024 election")
                if "harris" in article_text.lower() or "kamala" in article_text.lower():
                    entities.append("Kamala Harris 2024 election")
                if "president" in article_text.lower():
                    entities.append("Current US President 2024")

            # Step 2: Verify each entity/claim with web search
            logger.info("Step 2: Verifying entities and claims with web search")
            verification_results = []

            # Limit to most important entities to avoid too many searches
            search_entities = entities[:12] if len(entities) > 12 else entities

            for i, entity in enumerate(search_entities, 1):
                logger.info(f"Verifying entity/claim {i}/{len(search_entities)}: {entity}")

                # Search for information about this entity
                # Use more results for election-related queries
                num_results = 3 if any(keyword in entity.lower() for keyword in election_keywords) else 2
                search_result = self.web_search.search_and_summarize(entity, num_results=num_results)

                verification_results.append({
                    "entity": entity,
                    "search_results": search_result
                })

                # Minimal delay between searches
                time.sleep(0.1)

            # Step 3: Analyze the verification results
            logger.info("Step 3: Analyzing verification results")

            # Create a specialized prompt for factual checking with search results
            prompt = f"""
            TASK: Analyze this news article and assess its factual accuracy based on search results.

            ARTICLE TEXT:
            {article_text}

            ENTITIES/CLAIMS AND VERIFICATION RESULTS:
            """

            # Add verification results to the prompt
            for i, result in enumerate(verification_results, 1):
                prompt += f"\n--- ENTITY/CLAIM {i}: {result['entity']} ---\n"
                prompt += f"SEARCH RESULTS:\n{result['search_results']}\n"

            prompt += """
            INSTRUCTIONS:
            1. For each entity or claim, determine if it actually exists or is accurate based on the search results
            2. Pay special attention to:
               - People's names and their titles/positions - do they exist and hold those positions?
               - Organization names - do they exist as described?
               - Historical facts - are they accurately represented?
               - Dates and statistics - are they correct?
               - Events claimed to have happened - did they occur as described?
            3. Distinguish between major factual errors (that change the meaning of the article) and minor inaccuracies
            4. Determine what percentage of the article appears to be factually accurate

            CRITICAL FOR CURRENT EVENTS:
            - TRUST THE SEARCH RESULTS over any prior knowledge you may have
            - For claims about elections, political positions, or recent appointments, use ONLY the most recent information from search results
            - Pay special attention to dates mentioned in search results
            - For the 2024 US Presidential election, verify the current status based on search results
            - Check if people currently hold the positions or titles claimed in the article

            Consider these types of factual issues:
            - Non-existent people (e.g., "Pope Leo XIV" when no such pope exists)
            - Incorrect titles (e.g., claiming someone is President when they're not)
            - Events that never happened
            - Organizations that don't exist
            - Dates or statistics that are demonstrably wrong

            RESPONSE FORMAT:
            1. ENTITY VERIFICATION: List each entity/claim and whether it is VERIFIED, UNVERIFIED, or CONTRADICTED
            2. FACTUAL ASSESSMENT: Analyze the significance of any errors or inaccuracies found
            3. ACCURACY PERCENTAGE: Estimate what percentage of the article is factually accurate (e.g., "ACCURACY: 75%")
            4. CONCLUSION: Provide an overall assessment of the article's factual accuracy
            5. VERDICT: State "VERDICT: CONTAINS FACTUAL ERRORS" or "VERDICT: NO FACTUAL ERRORS FOUND"
            6. CONFIDENCE: State "CONFIDENCE: LOW/MEDIUM/HIGH" based on your certainty

            Remember to consider the significance of any errors in the context of the entire article. Major factual errors that change the meaning of the article are more important than minor inaccuracies.
            """

            # Generate analysis
            logger.info("Generating factual accuracy analysis with LLM")
            analysis = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract verdict
            contains_errors = "CONTAINS FACTUAL ERRORS" in analysis.upper()
            confidence = "LOW"

            # Default accuracy percentages
            real_percentage = 100 if not contains_errors else 0
            fake_percentage = 0 if not contains_errors else 100

            # Extract confidence
            if "CONFIDENCE: HIGH" in analysis.upper():
                confidence = "HIGH"
            elif "CONFIDENCE: MEDIUM" in analysis.upper():
                confidence = "MEDIUM"

            # Extract accuracy percentage
            accuracy_match = re.search(r'ACCURACY:\s*(\d+)%', analysis.upper())
            if accuracy_match:
                accuracy = int(accuracy_match.group(1))
                real_percentage = accuracy
                fake_percentage = 100 - accuracy
                logger.debug(f"Found accuracy percentage: {accuracy}%")

            # If no explicit accuracy percentage, try to find other indicators
            if not accuracy_match:
                # Look for percentage mentions
                percentage_match = re.search(r'(\d+)%\s*(?:ACCURATE|FACTUAL|CORRECT)', analysis.upper())
                if percentage_match:
                    accuracy = int(percentage_match.group(1))
                    real_percentage = accuracy
                    fake_percentage = 100 - accuracy
                    logger.debug(f"Found implied accuracy percentage: {accuracy}%")

            # Check for additional error indicators
            error_indicators = [
                "FACTUAL ERROR", "FACTUALLY INCORRECT", "INACCURATE",
                "DOES NOT EXIST", "NO SUCH PERSON", "FABRICATED",
                "INCORRECT TITLE", "NEVER HAPPENED", "CONTRADICTED",
                "OUTDATED INFORMATION", "NOT YET OCCURRED", "FUTURE EVENT",
                "HAS NOT HAPPENED YET", "HAS NOT TAKEN PLACE"
            ]

            for indicator in error_indicators:
                if indicator in analysis.upper():
                    contains_errors = True
                    logger.debug(f"Found error indicator: {indicator}")
                    # Don't break here - we want to check all indicators

            # Ensure percentages are reasonable based on verdict
            if contains_errors and real_percentage > 80:
                real_percentage = 70  # Cap real percentage if errors found
                fake_percentage = 30
                logger.debug("Capping real percentage to 70% due to factual errors")
            elif not contains_errors and real_percentage < 70:
                real_percentage = 80  # Minimum real percentage if no errors found
                fake_percentage = 20
                logger.debug("Setting minimum real percentage to 80% due to no factual errors")

            # Prepare result
            result = {
                "entities": entities,
                "verification_results": verification_results,
                "contains_errors": contains_errors,
                "analysis": analysis,
                "confidence": confidence,
                "real_percentage": real_percentage,
                "fake_percentage": fake_percentage
            }

            return result

        except Exception as e:
            log_exception(e, "check_factual_accuracy")
            logger.error(f"Error in check_factual_accuracy: {e}")

            # Return a default result with percentages
            return {
                "entities": [],
                "verification_results": [],
                "contains_errors": False,
                "analysis": f"Error performing factual check: {str(e)}",
                "confidence": "LOW",
                "real_percentage": 50,
                "fake_percentage": 50
            }

    def check_availability(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if the service is available, False otherwise
        """
        logger.info("Checking LLM service availability")

        try:
            available = self.ollama_client.check_availability()
            logger.info(f"LLM service availability: {available}")
            return available
        except Exception as e:
            log_exception(e, "check_availability")
            logger.error(f"Error checking LLM service availability: {e}")
            return False
