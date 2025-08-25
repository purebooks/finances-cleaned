/**
 * Universal File Reader - Handles all file formats and edge cases
 * Supports: CSV, JSON, TSV, Excel-like formats, different encodings, line endings
 */

class UniversalFileReader {
    constructor() {
        this.supportedExtensions = ['.csv', '.json', '.tsv', '.txt', '.xlsx', '.xls'];
        this.debug = true;
    }

    log(message) {
        if (this.debug) {
            console.log(`[UniversalFileReader] ${message}`);
        }
    }

    /**
     * Main entry point - reads any file and returns structured data
     */
    async readFile(file) {
        this.log(`Reading file: ${file.name} (${file.size} bytes, type: ${file.type})`);
        
        try {
            // Step 1: Detect file type and encoding
            const fileInfo = await this.analyzeFile(file);
            this.log(`File analysis: ${JSON.stringify(fileInfo, null, 2)}`);
            
            // Step 2: Read file content with appropriate method
            const rawContent = await this.readFileContent(file, fileInfo);
            this.log(`Raw content length: ${rawContent.length}`);
            
            // Step 3: Parse content based on detected format
            const structuredData = await this.parseContent(rawContent, fileInfo);
            this.log(`Parsed ${structuredData.length} records`);
            
            // Step 4: Validate and normalize data
            const validatedData = this.validateAndNormalize(structuredData);
            this.log(`Validated ${validatedData.length} records`);
            
            return {
                success: true,
                data: validatedData,
                metadata: {
                    originalFileName: file.name,
                    fileSize: file.size,
                    detectedFormat: fileInfo.format,
                    encoding: fileInfo.encoding,
                    lineEndings: fileInfo.lineEndings,
                    recordCount: validatedData.length
                }
            };
            
        } catch (error) {
            this.log(`Error reading file: ${error.message}`);
            return {
                success: false,
                error: error.message,
                data: [],
                metadata: {
                    originalFileName: file.name,
                    fileSize: file.size
                }
            };
        }
    }

    /**
     * Analyze file to detect format, encoding, and structure
     */
    async analyzeFile(file) {
        // Read first chunk for analysis
        const sampleSize = Math.min(file.size, 8192); // First 8KB
        const sampleBuffer = await this.readAsArrayBuffer(file.slice(0, sampleSize));
        const sampleBytes = new Uint8Array(sampleBuffer);
        
        // Detect encoding
        const encoding = this.detectEncoding(sampleBytes);
        
        // Convert sample to text for format detection
        const decoder = new TextDecoder(encoding);
        const sampleText = decoder.decode(sampleBytes);
        
        // Detect format
        const format = this.detectFormat(file.name, sampleText);
        
        // Detect line endings
        const lineEndings = this.detectLineEndings(sampleText);
        
        // Detect delimiter for CSV-like files
        const delimiter = this.detectDelimiter(sampleText, format);
        
        return {
            format,
            encoding,
            lineEndings,
            delimiter,
            hasBOM: this.detectBOM(sampleBytes)
        };
    }

    /**
     * Detect file encoding (UTF-8, UTF-16, etc.)
     */
    detectEncoding(bytes) {
        // Check for BOM
        if (bytes.length >= 3 && bytes[0] === 0xEF && bytes[1] === 0xBB && bytes[2] === 0xBF) {
            return 'utf-8';
        }
        if (bytes.length >= 2 && bytes[0] === 0xFF && bytes[1] === 0xFE) {
            return 'utf-16le';
        }
        if (bytes.length >= 2 && bytes[0] === 0xFE && bytes[1] === 0xFF) {
            return 'utf-16be';
        }
        
        // Default to UTF-8
        return 'utf-8';
    }

    /**
     * Detect BOM (Byte Order Mark)
     */
    detectBOM(bytes) {
        if (bytes.length >= 3 && bytes[0] === 0xEF && bytes[1] === 0xBB && bytes[2] === 0xBF) {
            return 'utf-8';
        }
        return null;
    }

    /**
     * Detect file format based on extension and content
     */
    detectFormat(fileName, sampleContent) {
        const ext = fileName.toLowerCase().substring(fileName.lastIndexOf('.'));
        
        // Extension-based detection
        if (ext === '.json') return 'json';
        if (ext === '.tsv') return 'tsv';
        if (ext === '.txt') return this.detectTextFormat(sampleContent);
        
        // Content-based detection for CSV
        if (ext === '.csv' || this.looksLikeCSV(sampleContent)) {
            return 'csv';
        }
        
        // Fallback
        return 'csv';
    }

    /**
     * Detect if content looks like CSV
     */
    looksLikeCSV(content) {
        const lines = content.split(/\r?\n/).slice(0, 5); // Check first 5 lines
        if (lines.length < 2) return false;
        
        // Check if lines have consistent comma count
        const commaCounts = lines.map(line => (line.match(/,/g) || []).length);
        const firstCount = commaCounts[0];
        const consistent = commaCounts.slice(1).every(count => Math.abs(count - firstCount) <= 1);
        
        return firstCount > 0 && consistent;
    }

    /**
     * Detect text format for .txt files
     */
    detectTextFormat(content) {
        if (content.trim().startsWith('{') || content.trim().startsWith('[')) {
            return 'json';
        }
        
        // Check for tab-separated
        const lines = content.split(/\r?\n/).slice(0, 3);
        const tabCounts = lines.map(line => (line.match(/\t/g) || []).length);
        if (tabCounts[0] > 0 && tabCounts.every(count => count === tabCounts[0])) {
            return 'tsv';
        }
        
        return 'csv';
    }

    /**
     * Detect line endings
     */
    detectLineEndings(content) {
        if (content.includes('\r\n')) return '\r\n';  // Windows
        if (content.includes('\n')) return '\n';      // Unix/Linux/Mac
        if (content.includes('\r')) return '\r';      // Old Mac
        return '\n'; // Default
    }

    /**
     * Detect delimiter for CSV-like files
     */
    detectDelimiter(content, format) {
        if (format === 'tsv') return '\t';
        
        const lines = content.split(/\r?\n/).slice(0, 5);
        const delimiters = [',', ';', '|', '\t'];
        
        let bestDelimiter = ',';
        let maxConsistency = 0;
        
        for (const delimiter of delimiters) {
            const counts = lines.map(line => (line.match(new RegExp(`\\${delimiter}`, 'g')) || []).length);
            if (counts[0] > 0) {
                const consistency = counts.filter(count => count === counts[0]).length / counts.length;
                if (consistency > maxConsistency) {
                    maxConsistency = consistency;
                    bestDelimiter = delimiter;
                }
            }
        }
        
        return bestDelimiter;
    }

    /**
     * Read file content with detected encoding
     */
    async readFileContent(file, fileInfo) {
        const arrayBuffer = await this.readAsArrayBuffer(file);
        const decoder = new TextDecoder(fileInfo.encoding);
        let content = decoder.decode(arrayBuffer);
        
        // Remove BOM if present
        if (fileInfo.hasBOM) {
            content = content.substring(1); // Remove BOM character
        }
        
        return content;
    }

    /**
     * Parse content based on detected format
     */
    async parseContent(content, fileInfo) {
        switch (fileInfo.format) {
            case 'json':
                return this.parseJSON(content);
            case 'csv':
                return this.parseCSV(content, fileInfo);
            case 'tsv':
                return this.parseTSV(content, fileInfo);
            default:
                return this.parseCSV(content, fileInfo); // Default fallback
        }
    }

    /**
     * Parse JSON content
     */
    parseJSON(content) {
        try {
            const parsed = JSON.parse(content.trim());
            
            // Handle different JSON structures
            if (Array.isArray(parsed)) {
                return parsed;
            } else if (parsed && typeof parsed === 'object') {
                // Single object - wrap in array
                return [parsed];
            } else {
                throw new Error('JSON must be an object or array');
            }
        } catch (error) {
            throw new Error(`Invalid JSON: ${error.message}`);
        }
    }

    /**
     * Parse CSV content with robust handling
     */
    parseCSV(content, fileInfo) {
        return this.parseDelimited(content, fileInfo.delimiter || ',', fileInfo);
    }

    /**
     * Parse TSV content
     */
    parseTSV(content, fileInfo) {
        return this.parseDelimited(content, '\t', fileInfo);
    }

    /**
     * Generic delimited file parser (CSV, TSV, etc.)
     */
    parseDelimited(content, delimiter, fileInfo) {
        // Normalize line endings
        const normalizedContent = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        
        // Split into lines and filter empty ones
        const lines = normalizedContent.split('\n').filter(line => line.trim().length > 0);
        
        if (lines.length === 0) {
            throw new Error('File appears to be empty');
        }
        
        if (lines.length < 2) {
            throw new Error(`File must have at least a header row and one data row. Found ${lines.length} lines.`);
        }

        // Parse header
        const headers = this.parseDelimitedLine(lines[0], delimiter);
        this.log(`Headers detected: ${JSON.stringify(headers)}`);
        
        if (headers.length === 0) {
            throw new Error('No valid headers found in first row');
        }

        // Parse data rows
        const data = [];
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.length === 0) continue; // Skip empty lines
            
            try {
                const values = this.parseDelimitedLine(line, delimiter);
                const row = {};
                
                // Map values to headers
                headers.forEach((header, index) => {
                    row[header] = values[index] || '';
                });
                
                data.push(row);
            } catch (error) {
                this.log(`Warning: Could not parse line ${i + 1}: ${line}. Error: ${error.message}`);
                // Continue processing other lines
            }
        }
        
        if (data.length === 0) {
            throw new Error('No valid data rows found');
        }
        
        return data;
    }

    /**
     * Parse a single delimited line handling quotes and escapes
     */
    parseDelimitedLine(line, delimiter) {
        const values = [];
        let current = '';
        let inQuotes = false;
        let i = 0;
        
        while (i < line.length) {
            const char = line[i];
            const nextChar = line[i + 1];
            
            if (char === '"') {
                if (inQuotes && nextChar === '"') {
                    // Escaped quote
                    current += '"';
                    i += 2;
                } else {
                    // Toggle quote state
                    inQuotes = !inQuotes;
                    i++;
                }
            } else if (char === delimiter && !inQuotes) {
                // End of field
                values.push(current.trim());
                current = '';
                i++;
            } else {
                current += char;
                i++;
            }
        }
        
        // Add final value
        values.push(current.trim());
        
        // Clean up quotes
        return values.map(value => {
            // Remove surrounding quotes if present
            if (value.startsWith('"') && value.endsWith('"')) {
                return value.slice(1, -1);
            }
            return value;
        });
    }

    /**
     * Validate and normalize parsed data
     */
    validateAndNormalize(data) {
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error('No valid data found');
        }
        
        return data.map((row, index) => {
            const normalizedRow = {};
            
            Object.keys(row).forEach(key => {
                // Normalize column names
                const normalizedKey = key.trim().toLowerCase();
                let value = row[key];
                
                // Normalize values
                if (typeof value === 'string') {
                    value = value.trim();
                    
                    // Try to convert numeric strings
                    if (normalizedKey.includes('amount') || normalizedKey.includes('price') || normalizedKey.includes('cost')) {
                        const numericValue = this.parseNumericValue(value);
                        if (numericValue !== null) {
                            value = numericValue;
                        }
                    }
                    
                    // Handle empty strings
                    if (value === '') {
                        value = null;
                    }
                }
                
                normalizedRow[key] = value; // Keep original key for compatibility
            });
            
            return normalizedRow;
        });
    }

    /**
     * Parse numeric values handling various formats
     */
    parseNumericValue(value) {
        if (!value || typeof value !== 'string') return null;
        
        // Remove common currency symbols and formatting
        const cleaned = value
            .replace(/[$€£¥₹]/g, '') // Currency symbols
            .replace(/[,\s]/g, '')   // Commas and spaces
            .replace(/[()]/g, '')    // Parentheses (accounting format)
            .trim();
        
        // Handle negative values in parentheses
        const isNegative = value.includes('(') && value.includes(')');
        
        const parsed = parseFloat(cleaned);
        if (isNaN(parsed)) return null;
        
        return isNegative ? -Math.abs(parsed) : parsed;
    }

    /**
     * Helper method to read file as ArrayBuffer
     */
    readAsArrayBuffer(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.onerror = e => reject(new Error(`File read error: ${e.target.error.message}`));
            reader.readAsArrayBuffer(file);
        });
    }
}

// Export for use in HTML files
window.UniversalFileReader = UniversalFileReader;