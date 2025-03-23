# Core Components

The core module provides fundamental building blocks and interfaces used throughout the probatus package.

## Overview

The core module establishes the architectural foundation for probatus components through abstract base classes and custom exceptions. This module defines consistent interfaces that ensure all probatus components follow similar patterns for fitting, computing results, and visualization.

### Key Components

#### Abstract Base Classes

##### BaseFitComputeClass
- Defines the standard interface for components that need to be fitted to data and compute results
- Implements a consistent pattern with three essential methods:
  - `fit()`: Trains the component on input data
  - `compute()`: Calculates results based on fitted state
  - `fit_compute()`: Combines fitting and computation in a single step
- Provides error checking to ensure methods are called in the correct sequence

##### BaseFitComputePlotClass
- Extends BaseFitComputeClass with visualization capabilities
- Adds the required `plot()` method for result visualization
- Used as the base class for components that need to generate visual insights
- Ensures consistent interface for all components with plotting functionality

#### Exceptions

##### NotFittedError
- Custom exception raised when a method requiring a fitted state is called prematurely
- Provides clear error messages indicating which operation failed
- Helps prevent incorrect usage of probatus components

### Core Design Principles

The core module implements several key design principles that are reflected throughout the probatus package:

1. **Consistent Interfaces**: All analytical components share a common interface pattern, making the library easier to learn and use.

2. **Method Chaining**: Base classes support method chaining (returning `self` from fit methods) for more concise code.

3. **Explicit State Validation**: Components explicitly check their fitted state before performing operations that require prior fitting.

4. **Separation of Concerns**: Clear separation between data fitting, result computation, and visualization.

5. **Extensibility**: Abstract base classes make it straightforward to add new components while maintaining consistent behavior.

These core components serve as the foundation for more specialized classes throughout probatus, ensuring that all parts of the library work together seamlessly.

## Implementation

::: probatus.core.base
::: probatus.core.exceptions
