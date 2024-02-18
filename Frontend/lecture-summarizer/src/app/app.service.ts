import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AppService {

  DEV = "http://127.0.0.1:5000"

  constructor(private httpClient: HttpClient) { }

  uploadData(data: any):Observable<any> {
    return this.httpClient.post<any>(this.DEV + '/upload', data);
  }
}
