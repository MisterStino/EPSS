# Apache Spark Setup Guide

This guide explains how to install and configure Apache Spark on your PC, including the Java requirements and verification steps.

## 1. Java Installation and Configuration

### 1.1. Which Java Version to Install
For Apache Spark, it is recommended to use **Java JDK 8 (Java 1.8)**.  
> **Reasoning:** Spark and its dependencies have been extensively tested with Java 8. Although some versions of Spark can work with Java 11, Java 8 is the most stable and widely supported version.

### 1.2. Installing Java JDK 8 (stijn: i use java 11.0.25)
1. **Download Java JDK 8:**
   - You can download it from the [Oracle Java SE Downloads page](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html) or use an OpenJDK version from [Adoptium](https://adoptium.net/).

2. **Install Java:**
   - Follow the installation instructions for your operating system (Windows, macOS, or Linux).

## Note stijn: check first if it works already, if not: but for java 11 our your java choice.

3. **Set the JAVA_HOME Environment Variable:**
   - **Windows:**
     1. Search for "Environment Variables" in the Start menu.
     2. Open "Edit the system environment variables."
     3. Under "System Properties" → "Advanced" → "Environment Variables," create a new variable `JAVA_HOME` with the path to your JDK installation (e.g., `C:\Program Files\Java\jdk1.8.0_321`).
   - **macOS/Linux:**
     Add the following lines to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):
     ```bash
     export JAVA_HOME=/path/to/jdk1.8.0_321
     export PATH=$JAVA_HOME/bin:$PATH
     ```

### 1.3. Checking Java Versions
- **Check the Default Java Version:**
  Open a terminal or command prompt and run:
  ```bash
  java -version
