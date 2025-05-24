import streamlit as st

def hide_sidebar_items():
    """
    Injects JavaScript and CSS to hide Model Explanation and Prediction from the sidebar.
    This function should be called at the beginning of each page.
    """
    # JavaScript to hide Model Explanation and Prediction from sidebar
    st.markdown("""
    <script>
    // Function to hide Model Explanation and Prediction from sidebar
    function hideSidebarItems() {
        // Find the sidebar navigation
        const sidebarNav = document.querySelector('[data-testid="stSidebarNav"]');
        if (sidebarNav) {
            // Method 1: Hide by position (3rd and 4th items)
            const navItems = sidebarNav.querySelectorAll('li');
            if (navItems.length >= 4) {
                // Hide the 3rd and 4th items (Model Explanation and Prediction)
                if (navItems[2]) navItems[2].style.display = 'none';
                if (navItems[3]) navItems[3].style.display = 'none';
            }
            
            // Method 2: Hide by text content
            const allLinks = sidebarNav.querySelectorAll('li a');
            allLinks.forEach(link => {
                // Check if the link text or href contains Model Explanation or Prediction
                const linkText = link.textContent.trim();
                const href = link.getAttribute('href') || '';
                
                if (linkText.includes('Model Explanation') ||
                    linkText.includes('Prediction') ||
                    href.includes('Model_Explanation') ||
                    href.includes('Prediction')) {
                    // Find the parent li element and hide it
                    const parentLi = link.closest('li');
                    if (parentLi) {
                        parentLi.style.display = 'none';
                        parentLi.style.visibility = 'hidden';
                        parentLi.style.height = '0';
                        parentLi.style.overflow = 'hidden';
                        parentLi.style.margin = '0';
                        parentLi.style.padding = '0';
                    }
                    
                    // Also hide the link itself
                    link.style.display = 'none';
                    link.style.visibility = 'hidden';
                }
            });
            
            // Method 3: Direct DOM removal
            const itemsToRemove = [];
            navItems.forEach(item => {
                const itemText = item.textContent.trim();
                if (itemText.includes('Model Explanation') || itemText.includes('Prediction')) {
                    itemsToRemove.push(item);
                }
            });
            
            // Remove the items from the DOM
            itemsToRemove.forEach(item => {
                try {
                    item.parentNode.removeChild(item);
                } catch (e) {
                    // Fallback to hiding if removal fails
                    item.style.display = 'none';
                }
            });
        }
    }

    // Function to rename "app" to "Home" in the sidebar
    function renameAppToHome() {
        // Find the sidebar navigation
        const sidebarNav = document.querySelector('[data-testid="stSidebarNav"]');
        if (sidebarNav) {
            // Find the first link which is typically the "app" link
            const firstLink = sidebarNav.querySelector('li:first-child a p');
            if (firstLink && firstLink.textContent.trim() === 'app') {
                // Change the text to "Home"
                firstLink.textContent = 'Home';
            }
            
            // Also try to find links by href
            const homeLinks = sidebarNav.querySelectorAll('a[href="/"], a[href="/app"]');
            homeLinks.forEach(link => {
                // Find any span or p elements inside this link
                const textElements = link.querySelectorAll('span, p');
                textElements.forEach(element => {
                    if (element.textContent.trim() === 'app') {
                        element.textContent = 'Home';
                    }
                });
            });
        }
    }

    // Run when the page loads
    document.addEventListener('DOMContentLoaded', function() {
        // Initial run
        hideSidebarItems();
        renameAppToHome();
        
        // Run again after a short delay to catch elements that might load later
        setTimeout(hideSidebarItems, 100);
        setTimeout(hideSidebarItems, 500);
        setTimeout(hideSidebarItems, 1000);
        
        // Set up an interval to keep checking
        const interval = setInterval(function() {
            hideSidebarItems();
            renameAppToHome();
        }, 1000);
        
        // Clear the interval after 30 seconds
        setTimeout(() => {
            clearInterval(interval);
        }, 30000);
    });

    // Use MutationObserver to detect DOM changes
    const observer = new MutationObserver(function(mutations) {
        // Run our functions when DOM changes
        hideSidebarItems();
        renameAppToHome();
        
        // Check if any mutations affected the sidebar navigation
        const sidebarMutation = mutations.some(mutation => {
            return mutation.target.closest && mutation.target.closest('[data-testid="stSidebarNav"]');
        });
        
        // If sidebar was affected, make sure to run our functions again
        if (sidebarMutation) {
            setTimeout(hideSidebarItems, 100);
            setTimeout(renameAppToHome, 100);
        }
    });

    // Start observing with a comprehensive configuration
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true
    });
    </script>
    
    <style>
    /* Hide Model Explanation and Prediction from sidebar - comprehensive approach */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:nth-child(3),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:nth-child(4),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="Model_Explanation"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="Prediction"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(a[href*="Model_Explanation"]),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(a[href*="Prediction"]),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="2_Model_Explanation"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="3_Prediction"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(a[href*="2_Model_Explanation"]),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(a[href*="3_Prediction"]) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        position: absolute !important;
        left: -9999px !important;
    }
    </style>
    """, unsafe_allow_html=True)
