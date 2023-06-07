package io.mosip.extractor.exception;

import io.mosip.kernel.core.exception.BaseUncheckedException;

/**
 * Custom Exception Class in case of error occurred in services.
 * 
 * @see io.mosip.kernel.core.exception.BaseUncheckedException
 * @author Janardhan B S
 * @since 1.0.0
 */
public class ExtractorException extends BaseUncheckedException {
	/**
	 * Generated serial version id
	 */

	/**
	 * Constructor the initialize Handler exception
	 * 
	 * @param errorCode    The error code for this exception
	 * @param errorMessage The error message for this exception
	 */
	public ExtractorException(String errorCode, String errorMessage) {
		super(errorCode, errorMessage);
	}

	/**
	 * Constructor the initialize Handler exception
	 * 
	 * @param errorCode    The error code for this exception
	 * @param errorMessage The error message for this exception
	 * @param rootCause    the specified cause
	 */
	public ExtractorException(String errorCode, String errorMessage, Throwable rootCause) {
		super(errorCode, errorMessage, rootCause);
	}
}