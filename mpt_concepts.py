import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def explain_mpt_concepts():
    """Explain the key concepts of Modern Portfolio Theory with visualizations."""
    print("\n" + "=" * 80)
    print(" MODERN PORTFOLIO THEORY (MPT) CONCEPTS ".center(80, "="))
    print("=" * 80 + "\n")
    
    print("Modern Portfolio Theory (MPT) is a framework for constructing an investment")
    print("portfolio that maximizes expected return for a given level of risk.")
    print("Here are the key concepts of MPT:\n")
    
    print("1. EXPECTED RETURN")
    print("   - The weighted average of the expected returns of the assets in the portfolio")
    print("   - Formula: E(Rp) = Σ (wi * E(Ri)) for all assets i")
    print("   - Where: wi = weight of asset i, E(Ri) = expected return of asset i\n")
    
    print("2. PORTFOLIO RISK (VOLATILITY)")
    print("   - Measured by the standard deviation of portfolio returns")
    print("   - Accounts for correlations between assets")
    print("   - Formula: σp = √(Σ Σ wi wj σi σj ρij) for all assets i,j")
    print("   - Where: σi = volatility of asset i, ρij = correlation between assets i and j\n")
    
    print("3. DIVERSIFICATION")
    print("   - Combining assets with low correlations reduces portfolio risk")
    print("   - The key insight of MPT: portfolio risk can be lower than the weighted")
    print("     average of individual asset risks\n")
    
    print("4. EFFICIENT FRONTIER")
    print("   - The set of optimal portfolios that offer the highest expected return")
    print("     for a given level of risk")
    print("   - Every portfolio below the frontier is suboptimal\n")
    
    print("5. CAPITAL ALLOCATION LINE (CAL)")
    print("   - Represents portfolios formed by combining the risk-free asset")
    print("     with a risky portfolio")
    print("   - The slope of the CAL is the Sharpe ratio\n")
    
    print("6. SHARPE RATIO")
    print("   - Measures excess return per unit of risk")
    print("   - Formula: (E(Rp) - Rf) / σp")
    print("   - Where: Rf = risk-free rate\n")
    
    print("7. TANGENCY PORTFOLIO")
    print("   - The portfolio on the efficient frontier that has the highest Sharpe ratio")
    print("   - Represents the optimal risky portfolio to combine with the risk-free asset\n")

def visualize_diversification():
    """Visualize the concept of diversification in MPT."""
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Define parameters
    returns = np.linspace(0, 0.2, 100)  # Expected returns from 0% to 20%
    
    # Plot individual assets
    plt.scatter(0.25, 0.12, color='red', s=100, label='Asset A')
    plt.scatter(0.15, 0.08, color='blue', s=100, label='Asset B')
    
    # Plot the diversification benefit curve (simplified)
    correlations = [-1, -0.5, 0, 0.5, 1]
    colors = ['green', 'limegreen', 'orange', 'salmon', 'red']
    
    for i, corr in enumerate(correlations):
        # Calculate portfolio risk for different weights
        weights = np.linspace(0, 1, 100)
        port_returns = weights * 0.12 + (1 - weights) * 0.08
        port_risks = np.sqrt(weights**2 * 0.25**2 + (1 - weights)**2 * 0.15**2 + 
                            2 * weights * (1 - weights) * 0.25 * 0.15 * corr)
        
        # Plot the curve
        plt.plot(port_risks, port_returns, color=colors[i], 
                 label=f'Correlation = {corr}')
    
    # Add labels and legend
    plt.title('The Power of Diversification in Modern Portfolio Theory')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('diversification_concept.png')
    plt.close()
    
    print("\nDiversification concept visualization saved to diversification_concept.png")
    print("This figure shows how combining two assets with different correlations")
    print("affects the risk-return profile of the portfolio.")
    print("When correlation is negative, the diversification benefit is strongest,")
    print("allowing for portfolios with lower risk than either individual asset.")

def visualize_efficient_frontier():
    """Visualize the efficient frontier and capital allocation line."""
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Define parameters for the efficient frontier (simplified elliptical shape)
    def confidence_ellipse(x_center, y_center, width, height, ax, n_std=3.0, **kwargs):
        """Create a plot of the covariance confidence ellipse."""
        pearson = 0.8  # Correlation between x and y
        
        # Using a special case to obtain the eigenvalues of the covariance matrix
        ell_radius_x = width / 2
        ell_radius_y = height / 2
        ellipse = Ellipse((x_center, y_center), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor='none', **kwargs)
        
        # Calculating the standard deviation of x from the square root of the variance
        scale_x = np.sqrt(ell_radius_x**2)
        mean_x = x_center
        
        # Calculating the standard deviation of y from the square root of the variance
        scale_y = np.sqrt(ell_radius_y**2)
        mean_y = y_center
        
        # Get the transformation
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    
    # Plot the feasible set of portfolios
    ax = plt.gca()
    confidence_ellipse(0.2, 0.1, 0.3, 0.15, ax, alpha=0.1, color='gray', zorder=0)
    
    # Plot the efficient frontier
    x = np.linspace(0.1, 0.35, 100)
    y = 0.05 + 0.5 * (x - 0.1) - 0.5 * (x - 0.1)**2
    plt.plot(x, y, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot the capital allocation line
    risk_free_rate = 0.02
    x_tangent = 0.2
    y_tangent = 0.05 + 0.5 * (x_tangent - 0.1) - 0.5 * (x_tangent - 0.1)**2
    sharpe_ratio = (y_tangent - risk_free_rate) / x_tangent
    
    x_cal = np.linspace(0, 0.4, 100)
    y_cal = risk_free_rate + sharpe_ratio * x_cal
    plt.plot(x_cal, y_cal, 'r-', linewidth=2, label='Capital Allocation Line')
    
    # Plot the risk-free asset
    plt.scatter(0, risk_free_rate, color='green', s=100, label='Risk-Free Asset')
    
    # Plot the tangency portfolio
    plt.scatter(x_tangent, y_tangent, color='red', s=100, label='Tangency Portfolio')
    
    # Add labels and legend
    plt.title('Efficient Frontier and Capital Allocation Line')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('efficient_frontier_concept.png')
    plt.close()
    
    print("\nEfficient frontier concept visualization saved to efficient_frontier_concept.png")
    print("This figure illustrates:")
    print("- The efficient frontier: optimal portfolios with maximum return for a given risk")
    print("- The capital allocation line: combinations of the risk-free asset and the tangency portfolio")
    print("- The tangency portfolio: the portfolio on the efficient frontier with the highest Sharpe ratio")

def visualize_portfolio_weights():
    """Visualize how portfolio weights affect risk and return."""
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define parameters
    weights = np.linspace(0, 1, 100)
    
    # Asset parameters
    returns_a = 0.15  # 15% return for asset A
    risk_a = 0.25     # 25% volatility for asset A
    returns_b = 0.08  # 8% return for asset B
    risk_b = 0.12     # 12% volatility for asset B
    correlation = 0.3 # 30% correlation between assets
    
    # Calculate portfolio metrics for different weights
    port_returns = weights * returns_a + (1 - weights) * returns_b
    port_risks = np.sqrt(weights**2 * risk_a**2 + (1 - weights)**2 * risk_b**2 + 
                        2 * weights * (1 - weights) * risk_a * risk_b * correlation)
    
    # Plot portfolio return vs. weight
    ax1.plot(weights, port_returns, 'b-', linewidth=2)
    ax1.set_title('Portfolio Return vs. Weight in Asset A')
    ax1.set_xlabel('Weight in Asset A')
    ax1.set_ylabel('Expected Return')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=returns_a, color='r', linestyle='--', alpha=0.5, label='Asset A Return')
    ax1.axhline(y=returns_b, color='g', linestyle='--', alpha=0.5, label='Asset B Return')
    ax1.legend()
    
    # Plot portfolio risk vs. weight
    ax2.plot(weights, port_risks, 'r-', linewidth=2)
    ax2.set_title('Portfolio Risk vs. Weight in Asset A')
    ax2.set_xlabel('Weight in Asset A')
    ax2.set_ylabel('Portfolio Risk (Standard Deviation)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=risk_a, color='r', linestyle='--', alpha=0.5, label='Asset A Risk')
    ax2.axhline(y=risk_b, color='g', linestyle='--', alpha=0.5, label='Asset B Risk')
    
    # Find and mark the minimum risk portfolio
    min_risk_idx = np.argmin(port_risks)
    min_risk_weight = weights[min_risk_idx]
    min_risk = port_risks[min_risk_idx]
    
    ax2.scatter(min_risk_weight, min_risk, color='blue', s=100, 
               label=f'Minimum Risk Portfolio (w={min_risk_weight:.2f})')
    ax2.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('portfolio_weights_concept.png')
    plt.close()
    
    print("\nPortfolio weights concept visualization saved to portfolio_weights_concept.png")
    print("This figure shows:")
    print("- Left: How the portfolio's expected return changes with different weights")
    print("- Right: How the portfolio's risk changes with different weights")
    print(f"- The minimum risk portfolio occurs at {min_risk_weight:.2f} weight in Asset A")
    print("- Note that the minimum risk can be lower than either individual asset's risk")
    print("  due to the diversification effect")

def main():
    """Main function to explain MPT concepts with visualizations."""
    try:
        # Explain MPT concepts
        explain_mpt_concepts()
        
        # Visualize diversification
        visualize_diversification()
        
        # Visualize efficient frontier
        visualize_efficient_frontier()
        
        # Visualize portfolio weights
        visualize_portfolio_weights()
        
        print("\nMPT concepts explanation and visualizations completed successfully!")
        print("These visualizations will help you understand the key concepts of Modern Portfolio Theory")
        print("and how they apply to the portfolio optimization performed in this project.")
    except Exception as e:
        print(f"Error in MPT concepts explanation: {e}")

if __name__ == "__main__":
    main()